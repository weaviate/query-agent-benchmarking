import inspect
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
import yaml

import weaviate.collections.classes.config as wvcc

from .spec import DatasetSpec
from query_agent_benchmarking.internal.adapters.database.naming import pascalize_name
from query_agent_benchmarking.internal.adapters.clients.provider_headers import parse_embedding_model

TEXT = wvcc.DataType.TEXT
BLOB = wvcc.DataType.BLOB
INT = wvcc.DataType.INT
FIELD = wvcc.Tokenization.FIELD

def dataset_id_property() -> wvcc.Property:
    """Standard dataset_id property for filtering."""
    return wvcc.Property(
        name="dataset_id",
        data_type=TEXT,
        index_searchable=False,
        index_filterable=True,
        skip_vectorization=True,
        tokenization=FIELD,
    )


class DatasetSpecBuilder:
    """Fluent builder for DatasetSpec."""

    def __init__(self, pattern: str, config_path: Optional[str | Path] = None):
        self._pattern = pattern
        self._name_fn: Optional[Callable[[str], str]] = None
        self._properties: list[wvcc.Property] = []
        self._vector_config: Any = None
        self._field_mappings: dict[str, str] = {}
        self._id_field: str = "dataset_id"
        self._custom_mapper: Optional[Callable[[Mapping[str, Any]], dict[str, Any]]] = None
        self._multi_tenancy_config: Any = None
        self._tenant_id_field: Optional[str] = None
        self._config: dict[str, Any] = self._load_config(config_path)

    def _load_config(self, path: Optional[str | Path] = None) -> dict[str, Any]:
        """Load configuration from a YAML file."""
        if path is None:
            # Resolve relative to this file's directory, not the current working directory
            path = Path(__file__).parent.parent.parent / "config" / "database_loader_config.yml"
        else:
            path = Path(path)

        if not path.exists():
            return {}

        with open(path) as f:
            return yaml.safe_load(f) or {}

    def with_name(self, name_fn: Callable[[str], str]) -> "DatasetSpecBuilder":
        """Set a custom collection name function."""
        self._name_fn = name_fn
        return self

    def with_static_name(self, name: str) -> "DatasetSpecBuilder":
        """Set a static collection name."""
        self._name_fn = lambda _: name
        return self

    def with_prefix_name(
        self,
        prefix: str,
        split_char: str = "/",
        split_index: int = 1,
    ) -> "DatasetSpecBuilder":
        """
        Set a name derived from the dataset name.
        
        E.g., "beir/scifact" with prefix="Beir" -> "BeirScifact"
        """
        self._name_fn = lambda d: f"{prefix}{pascalize_name(d.split(split_char)[split_index])}"
        return self

    def with_text_property(
        self,
        name: str,
        source_field: Optional[str] = None,
        searchable: bool = True,
        filterable: bool = True,
    ) -> "DatasetSpecBuilder":
        """Add a text property."""
        self._properties.append(
            wvcc.Property(
                name=name,
                data_type=TEXT,
                index_searchable=searchable,
                index_filterable=filterable,
            )
        )
        self._field_mappings[name] = source_field or name
        return self

    def with_blob_property(
        self,
        name: str,
        source_field: Optional[str] = None,
    ) -> "DatasetSpecBuilder":
        """Add a blob property for binary data."""
        self._properties.append(wvcc.Property(name=name, data_type=BLOB))
        self._field_mappings[name] = source_field or name
        return self

    def with_int_property(
        self,
        name: str,
        source_field: Optional[str] = None,
        filterable: bool = True,
    ) -> "DatasetSpecBuilder":
        """Add an integer property."""
        self._properties.append(
            wvcc.Property(name=name, data_type=INT, index_filterable=filterable)
        )
        self._field_mappings[name] = source_field or name
        return self

    def with_dataset_id(self, source_field: str = "dataset_id") -> "DatasetSpecBuilder":
        """Add the standard dataset_id property for filtering."""
        self._properties.append(dataset_id_property())
        self._id_field = source_field
        return self

    # --- Vectorizers ---

    def _parse_embedding_model(self, embedding_model: str) -> tuple[str, str]:
        """Parse embedding model string into provider and model name."""
        return parse_embedding_model(embedding_model)

    def _provider_suffix(self, provider: str) -> str:
        """Normalize provider labels for vector names."""
        normalized = provider.strip().lower()
        if normalized in {"voyage", "voyage-ai"}:
            return "voyageai"
        if normalized in {"gemini", "google-gemini", "google_gemini"}:
            return "google"
        return normalized

    def _qualify_vector_name_with_provider(
        self,
        name: Optional[str],
        provider: str,
    ) -> Optional[str]:
        """Append provider suffix to a base vector name."""
        if name is None:
            return None
        return f"{name}_{self._provider_suffix(provider)}"

    def _get_vectorizer_for_provider(self, provider: str, model: str) -> Any:
        """
        Get the appropriate vectorizer config based on provider.
        
        Args:
            provider: The provider name ("cohere", "weaviate", or "voyageai")
            model: The model name to use
        
        Returns:
            Vectorizer configuration object
        """
        if provider == "weaviate":
            return wvcc.Configure.Vectors.text2vec_weaviate(
                model=model,
                vectorize_collection_name=False,
            )
        elif provider == "cohere":
            return wvcc.Configure.Vectors.text2vec_cohere(model=model)
        elif provider == "voyageai":
            return wvcc.Configure.Vectors.text2vec_voyageai(model=model)
        elif provider == "google":
            return wvcc.Configure.Vectors.multi2vec_google_gemini(model=model)
        else:
            raise ValueError(
                f"Unsupported embedding provider: '{provider}'. "
                f"Supported providers: ['weaviate', 'cohere', 'voyageai', 'google']"
            )

    def _append_vector_config(self, cfg: Any) -> None:
        """Append a vector config to a named-vectors list."""
        if self._vector_config is None:
            self._vector_config = [cfg]
            return
        if isinstance(self._vector_config, list):
            new_name = getattr(cfg, "name", None)
            if new_name:
                existing_names = {
                    getattr(existing_cfg, "name", None)
                    for existing_cfg in self._vector_config
                }
                if new_name in existing_names:
                    raise ValueError(
                        f"Named vector '{new_name}' is already configured. "
                        "Vector names must be unique within a collection."
                    )
            self._vector_config.append(cfg)
            return
        raise ValueError(
            "Cannot append a named vector config because vector_config is already set "
            "to a single (unnamed) vectorizer. Use named vector configs consistently "
            "(i.e., call with_text2vec/with_multi2vec with `name=`) or start a new builder."
        )

    def _set_or_append_vector_config(self, cfg: Any, name: Optional[str]) -> None:
        """Set unnamed vector config or append a named vector config."""
        if name is None:
            if isinstance(self._vector_config, list):
                raise ValueError(
                    "Cannot set an unnamed vectorizer when named vectors are already configured. "
                    "Pass `name=` instead."
                )
            self._vector_config = cfg
            return
        self._append_vector_config(cfg)

    def with_text2vec(
        self,
        name: Optional[str] = None,
        source_properties: Optional[list[str]] = None,
        embedding_model: Optional[str] = None,
        name_by_provider: bool = False,
    ) -> "DatasetSpecBuilder":
        """
        Configure a text2vec vectorizer.

        - If called with no args, configures a single unnamed vectorizer (backward compatible).
        - If called with `name` and `source_properties`, appends a named vector config, enabling
          multiple independent named vectors (e.g., `.with_text2vec(...).with_text2vec(...)`).

        Reads the 'text_embedding_model' config value (e.g., "cohere/embed-4")
        and selects the appropriate vectorizer provider.
        """
        if (name is None) != (source_properties is None):
            raise ValueError("with_text2vec: `name` and `source_properties` must be provided together")

        cfg_model = embedding_model or self._config.get("text_embedding_model")

        # Unnamed, single-vector mode (legacy)
        if name is None and source_properties is None:
            if isinstance(self._vector_config, list):
                raise ValueError(
                    "with_text2vec(): cannot set an unnamed vectorizer when named vectors are already configured. "
                    "Pass `name=` and `source_properties=` instead."
                )
            if not cfg_model:
                self._vector_config = wvcc.Configure.Vectors.text2vec_weaviate(
                    vectorize_collection_name=False,
                )
                return self
            provider, model_name = self._parse_embedding_model(cfg_model)
            self._vector_config = self._get_vectorizer_for_provider(provider, model_name)
            return self

        # Named-vectors mode
        if not cfg_model:
            vector_name = (
                self._qualify_vector_name_with_provider(name, "weaviate")
                if name_by_provider
                else name
            )
            # Default to Weaviate vectorizer with explicit name/source_properties
            self._append_vector_config(
                wvcc.Configure.Vectors.text2vec_weaviate(
                    name=vector_name,
                    source_properties=source_properties,
                    vectorize_collection_name=False,
                )
            )
            return self

        provider, model_name = self._parse_embedding_model(cfg_model)
        vector_name = (
            self._qualify_vector_name_with_provider(name, provider)
            if name_by_provider
            else name
        )

        if provider == "weaviate":
            cfg = wvcc.Configure.Vectors.text2vec_weaviate(
                name=vector_name,
                model=model_name,
                source_properties=source_properties,
                vectorize_collection_name=False,
            )
        elif provider == "cohere":
            cfg = wvcc.Configure.Vectors.text2vec_cohere(
                name=vector_name,
                model=model_name,
                source_properties=source_properties,
            )
        elif provider == "voyageai":
            cfg = wvcc.Configure.Vectors.text2vec_voyageai(
                name=vector_name,
                model=model_name,
                source_properties=source_properties,
            )
        elif provider == "google":
            cfg = wvcc.Configure.Vectors.multi2vec_google_gemini(
                name=vector_name,
                model=model_name,
                text_fields=source_properties,
            )
        else:
            raise ValueError(
                f"Unsupported embedding provider: '{provider}'. "
                f"Supported providers: ['weaviate', 'cohere', 'voyageai', 'google']"
            )

        self._append_vector_config(cfg)
        return self

    def with_text2vec_provider_named(
        self,
        source_properties: list[str],
        base_name: str = "text_content",
        embedding_models: Optional[list[str]] = None,
    ) -> "DatasetSpecBuilder":
        """
        Configure one or more provider-qualified text named vectors.

        Naming convention:
        - Single provider: `<base_name>_<provider>`
          Example: `text_content_weaviate`
        - Repeated provider in a single collection: `<base_name>_<provider>_<n>`
          Example: `text_content_cohere_2`
        """
        configured_models = embedding_models
        if configured_models is None:
            configured_models = self._config.get("text_embedding_models")

        if configured_models is None:
            return self.with_text2vec(
                name=base_name,
                source_properties=source_properties,
                name_by_provider=True,
            )
        if not isinstance(configured_models, list) or not configured_models:
            raise ValueError(
                "text_embedding_models must be a non-empty list when provided."
            )

        provider_totals: dict[str, int] = {}
        parsed_models: list[tuple[str, str]] = []
        for model in configured_models:
            if not isinstance(model, str) or not model.strip():
                raise ValueError(
                    "Every value in text_embedding_models must be a non-empty string."
                )
            provider, _ = self._parse_embedding_model(model.strip())
            provider_label = self._provider_suffix(provider)
            provider_totals[provider_label] = provider_totals.get(provider_label, 0) + 1
            parsed_models.append((model.strip(), provider_label))

        provider_seen: dict[str, int] = {}
        for model, provider_label in parsed_models:
            provider_seen[provider_label] = provider_seen.get(provider_label, 0) + 1
            suffix = provider_label
            if provider_totals[provider_label] > 1:
                suffix = f"{provider_label}_{provider_seen[provider_label]}"
            vector_name = f"{base_name}_{suffix}"
            self.with_text2vec(
                name=vector_name,
                source_properties=source_properties,
                embedding_model=model,
            )
        return self

    def with_multi2vec(
        self,
        image_field: str,
        name: Optional[str] = None,
        default_model: str = "ModernVBERT/colmodernvbert",
        embedding_model: Optional[str] = None,
        name_by_provider: bool = False,
    ) -> "DatasetSpecBuilder":
        """
        Configure a multi2vec image vectorizer via provider dispatch.

        Provider-specific config creation is intentionally isolated in
        `_build_multi2vec_config(...)` so new `multi2vec_X` providers can be added
        in one place.

        - If called with `name`, appends a named vector config (named-vectors mode).
        - If called without `name`, configures a single unnamed vectorizer (backward compatible).

        Reads `image_embedding_model` (or falls back to `text_embedding_model`).
        Example: `"weaviate/ModernVBERT/colmodernvbert"`.
        
        Args:
            image_field: The property name containing the image blob
            name: Optional named vector name (enables multiple vectors per collection)
            default_model: Default model if none specified in config
            embedding_model: Optional override for the model string ("provider/model")
        """
        provider, model_name = self._resolve_multi2vec_provider_model(
            embedding_model=embedding_model,
            default_model=default_model,
        )
        vector_name = (
            self._qualify_vector_name_with_provider(name, provider)
            if name_by_provider
            else name
        )
        cfg = self._build_multi2vec_config(
            provider=provider,
            image_field=image_field,
            model=model_name,
            name=vector_name,
        )
        self._set_or_append_vector_config(cfg, name=vector_name)
        return self

    def with_image2vec(
        self,
        image_field: str,
        name: Optional[str] = None,
        name_by_provider: bool = False,
    ) -> "DatasetSpecBuilder":
        """
        Configure the `img2vec-neural` image vectorizer.

        This wraps `Configure.Vectors.img2vec_neural(...)` and supports both
        unnamed and named-vector modes.

        Args:
            image_field: Blob property containing base64-encoded image data.
            name: Optional named vector name.
        """
        vector_name = (
            self._qualify_vector_name_with_provider(name, "weaviate")
            if name_by_provider
            else name
        )
        cfg = wvcc.Configure.Vectors.img2vec_neural(
            name=vector_name,
            image_fields=[image_field],
        )
        self._set_or_append_vector_config(cfg, name=vector_name)
        return self

    def with_multi2vec_weaviate(
        self,
        image_field: str,
        model: str = "ModernVBERT/colmodernvbert",
        name: Optional[str] = None,
        name_by_provider: bool = False,
    ) -> "DatasetSpecBuilder":
        """Convenience wrapper for the Weaviate multi2vec provider."""
        vector_name = (
            self._qualify_vector_name_with_provider(name, "weaviate")
            if name_by_provider
            else name
        )
        cfg = self._build_multi2vec_config(
            provider="weaviate",
            image_field=image_field,
            model=model,
            name=vector_name,
        )
        self._set_or_append_vector_config(cfg, name=vector_name)
        return self

    def _resolve_multi2vec_provider_model(
        self,
        embedding_model: Optional[str],
        default_model: str,
    ) -> tuple[str, str]:
        """Resolve provider/model for multi2vec configuration."""
        text_embedding_models = self._config.get("text_embedding_models")
        first_text_embedding_model = (
            text_embedding_models[0]
            if isinstance(text_embedding_models, list) and text_embedding_models
            else None
        )
        cfg_model = (
            embedding_model
            or self._config.get("image_embedding_model")
            or self._config.get("text_embedding_model")
            or first_text_embedding_model
        )
        if not cfg_model:
            return "weaviate", default_model

        provider, model_name = self._parse_embedding_model(cfg_model)
        return provider, model_name or default_model

    def _multi2vec_supports_muvera(self, provider: str) -> bool:
        """Return whether a multi2vec provider supports MUVERA encoding."""
        return provider == "weaviate"

    def _build_muvera_kwargs(self, provider: str) -> dict[str, Any]:
        """Build MUVERA kwargs if enabled in config for a compatible provider."""
        if not self._config.get("use_MUVERA_encoding"):
            return {}
        if not self._multi2vec_supports_muvera(provider):
            raise ValueError(
                f"use_MUVERA_encoding=True is not supported for provider '{provider}'. "
                "Disable MUVERA or use a provider that supports it."
            )
        return {
            "encoding": wvcc.Configure.VectorIndex.MultiVector.Encoding.muvera(
                ksim=self._config.get("ksim", 4),
                dprojections=self._config.get("dprojections", 16),
                repetitions=self._config.get("repetitions", 10),
            ),
            "vector_index_config": wvcc.Configure.VectorIndex.hnsw(
                ef=self._config.get("ef", 500),
            ),
        }

    def _build_multi2vec_weaviate_config(
        self,
        image_field: str,
        model: str,
        name: Optional[str],
    ) -> Any:
        """Build multi2vec config for the Weaviate provider."""
        constructor = getattr(wvcc.Configure.MultiVectors, "multi2vec_weaviate", None)
        if constructor is None:
            constructor = getattr(wvcc.Configure.Vectors, "multi2vec_weaviate", None)
        if constructor is None:
            raise ValueError(
                "This Weaviate client version does not expose multi2vec_weaviate."
            )

        kwargs: dict[str, Any] = {"name": name, "model": model}
        try:
            signature = inspect.signature(constructor)
        except (TypeError, ValueError):
            signature = None

        if signature and "image_fields" in signature.parameters:
            kwargs["image_fields"] = [image_field]
        elif signature and "image_field" in signature.parameters:
            kwargs["image_field"] = image_field
        else:
            # Default to modern parameter shape.
            kwargs["image_fields"] = [image_field]

        kwargs.update(self._build_muvera_kwargs(provider="weaviate"))
        try:
            return constructor(**kwargs)
        except TypeError as exc:
            # Compatibility fallback between client versions.
            error = str(exc)
            if "unexpected keyword argument 'image_fields'" in error:
                kwargs.pop("image_fields", None)
                kwargs["image_field"] = image_field
                return constructor(**kwargs)
            if "unexpected keyword argument 'image_field'" in error:
                kwargs.pop("image_field", None)
                kwargs["image_fields"] = [image_field]
                return constructor(**kwargs)
            raise

    def _build_multi2vec_cohere_config(
        self,
        image_field: str,
        model: str,
        name: Optional[str],
    ) -> Any:
        """Build multi2vec config for the Cohere provider."""
        constructor = getattr(wvcc.Configure.Vectors, "multi2vec_cohere", None)
        if constructor is None:
            raise ValueError(
                "This Weaviate client version does not expose multi2vec_cohere. "
                "Please upgrade your weaviate-client package."
            )
        return constructor(
            name=name,
            model=model,
            image_fields=[image_field],
        )

    def _build_multi2vec_google_gemini_config(
        self,
        image_field: str,
        model: str,
        name: Optional[str],
    ) -> Any:
        """Build multi2vec config for the Google Gemini provider."""
        constructor = getattr(wvcc.Configure.Vectors, "multi2vec_google_gemini", None)
        if constructor is None:
            raise ValueError(
                "This Weaviate client version does not expose multi2vec_google_gemini. "
                "Please upgrade your weaviate-client package."
            )
        return constructor(
            name=name,
            model=model,
            image_fields=[image_field],
        )

    def _build_multi2vec_config(
        self,
        provider: str,
        image_field: str,
        model: str,
        name: Optional[str],
    ) -> Any:
        """
        Build provider-specific multi2vec config.

        To add a new provider in the future, register it in `provider_builders`.
        """
        provider_builders: dict[str, Callable[[str, str, Optional[str]], Any]] = {
            "weaviate": self._build_multi2vec_weaviate_config,
            "cohere": self._build_multi2vec_cohere_config,
            "google": self._build_multi2vec_google_gemini_config,
        }
        builder = provider_builders.get(provider)
        if builder is None:
            supported = sorted(provider_builders.keys())
            raise ValueError(
                f"Unsupported multi2vec provider: '{provider}'. "
                f"Supported providers in this project: {supported}"
            )
        return builder(image_field, model, name)

    def with_multi_tenancy(
        self,
        tenant_id_field: str = "tenant_id",
    ) -> "DatasetSpecBuilder":
        """Enable multi-tenancy for this dataset.

        Args:
            tenant_id_field: Source field name used to group items by tenant.
        """
        from weaviate.collections.classes.config import Configure
        self._multi_tenancy_config = Configure.multi_tenancy(enabled=True, auto_tenant_creation=True)
        self._tenant_id_field = tenant_id_field
        return self

    def with_custom_vector_config(self, config: Any) -> "DatasetSpecBuilder":
        """Use a custom vector configuration."""
        self._vector_config = config
        return self

    # --- Custom mapping ---

    def with_custom_mapper(
        self,
        mapper: Callable[[Mapping[str, Any]], dict[str, Any]],
    ) -> "DatasetSpecBuilder":
        """Override the auto-generated property mapper."""
        self._custom_mapper = mapper
        return self

    # --- Build ---

    def build(self) -> DatasetSpec:
        """Build the final DatasetSpec."""
        if self._name_fn is None:
            raise ValueError("Name function is required")
        if self._vector_config is None:
            raise ValueError("Vector config is required")
        if not self._properties:
            raise ValueError("At least one property is required")

        item_to_props = self._custom_mapper or self._build_mapper()

        return DatasetSpec(
            pattern=self._pattern,
            name_fn=self._name_fn,
            properties=tuple(self._properties),
            vector_config=self._vector_config,
            item_to_props=item_to_props,
            multi_tenancy_config=self._multi_tenancy_config,
            tenant_id_field=self._tenant_id_field,
        )

    def _build_mapper(self) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
        """Build the property mapper from field mappings."""
        field_mappings = self._field_mappings.copy()
        id_field = self._id_field

        def mapper(item: Mapping[str, Any]) -> dict[str, Any]:
            result = {}
            for prop_name, source_field in field_mappings.items():
                if source_field in item:
                    result[prop_name] = item[source_field]
            if id_field in item:
                result["dataset_id"] = str(item[id_field])
            return result

        return mapper
