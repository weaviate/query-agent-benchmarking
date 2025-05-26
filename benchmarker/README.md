# Weaviate Query Agent Benchmarker

This repository provides a comprehensive benchmarking framework for evaluating Weaviate Query Agents and custom RAG (Retrieval-Augmented Generation) implementations. The system supports both synchronous and asynchronous execution, multiple datasets, and various evaluation metrics.

## 🏗️ Architecture Overview

The benchmarker is built around a modular architecture that separates concerns into distinct components:

- **Agent Management**: Unified interface for different agent types (Weaviate Query Agent, DSPy RAG variants)
- **Dataset Handling**: Support for multiple datasets with standardized loading and preprocessing
- **Database Operations**: Automated Weaviate collection management and data ingestion
- **Evaluation Framework**: Comprehensive metrics including IR metrics and LM-as-a-Judge evaluation
- **Async Support**: High-performance concurrent query execution with rate limiting

## 📁 Project Structure

```
src/
├── agent.py                    # Agent abstraction and builder
├── query_agent_benchmark.py    # Core benchmarking logic
├── benchmark-run.py            # Main execution script
├── config.yml                  # Configuration settings
├── database.py                 # Weaviate database management
├── dataset.py                  # Dataset loading and preprocessing
├── utils.py                    # Utility functions and result serialization
├── arena-run.py                # Arena-style comparison runner (placeholder)
├── dspy_rag/                   # DSPy RAG implementations
│   ├── rag_programs.py         # RAG pipeline variants
│   ├── rag_signatures.py       # DSPy interface definitions
│   └── dspy_rag_utils.py       # Search utilities for RAG
├── metrics/                    # Evaluation metrics
│   ├── ir_metrics.py           # Information retrieval metrics
│   └── lm_as_judge_agent.py    # LM-based answer quality evaluation
├── models/                     # Data models
│   └── weaviate_query.py       # Weaviate query structure definitions
└── results/                    # Output directory for benchmark results
```

## 🔧 Core Components

### 1. Agent Management (`agent.py`)

The `AgentBuilder` class provides a unified interface for different agent types:

**Supported Agents:**
- **Weaviate Query Agent**: Production-ready hosted agent from Weaviate
- **DSPy RAG Variants**: Custom implementations for research and comparison
  - `vanilla-rag`: Basic RAG with single search
  - `search-only`: Search without answer generation
  - `search-only-with-qw`: Search with query writing
  - `query-writer-rag`: Multi-query RAG with query decomposition

**Key Features:**
- Async/sync execution modes
- Automatic dataset-specific configuration
- Environment-based credential management
- Connection pooling and cleanup

### 2. Benchmarking Engine (`query_agent_benchmark.py`)

The core benchmarking logic supports both synchronous and asynchronous execution:

**Async Features:**
- Concurrent query processing with configurable batch sizes
- Rate limiting with exponential backoff
- Progress tracking and error handling
- Automatic retry logic for connection failures

**Evaluation Pipeline:**
- Query execution timing
- Source retrieval tracking
- Search/aggregation operation counting
- Comprehensive result analysis

### 3. Dataset Management (`dataset.py`)

Supports multiple datasets with standardized interfaces:

**Supported Datasets:**
- **Enron**: Email corpus for enterprise search scenarios
- **WixQA**: Knowledge base Q&A dataset
- **FreshStack**: Multi-domain technical documentation
  - Angular, Godot, LangChain, Laravel, YOLO subsets

**Features:**
- Automatic HuggingFace dataset loading
- Nugget-based evaluation support for complex queries
- Standardized ID mapping and metadata handling

### 4. Database Operations (`database.py`)

Automated Weaviate collection management:

**Capabilities:**
- Collection creation with appropriate schemas
- Batch data ingestion with progress tracking
- Dataset-specific property mapping
- Concurrent upload optimization

### 5. DSPy RAG Framework (`dspy_rag/`)

Custom RAG implementations using the DSPy framework:

**RAG Variants:**
- **VanillaRAG**: Single-query retrieval with answer generation
- **SearchQueryWriter**: Multi-query decomposition with aggregated retrieval
- **SearchOnlyRAG**: Retrieval-only for IR evaluation
- **SearchOnlyWithQueryWriter**: Query expansion without answer generation

**Components:**
- **Signatures** (`rag_signatures.py`): DSPy interface definitions for each pipeline stage
- **Programs** (`rag_programs.py`): Complete RAG pipeline implementations
- **Utils** (`dspy_rag_utils.py`): Weaviate search integration and result formatting

### 6. Evaluation Metrics (`metrics/`)

Comprehensive evaluation framework:

**Information Retrieval Metrics** (`ir_metrics.py`):
- Standard recall calculation
- Nugget-based evaluation for complex queries
- Support for multi-relevant document scenarios

**LM-as-a-Judge** (`lm_as_judge_agent.py`):
- Automated answer quality assessment
- 1-5 scale rating with reasoning
- Retry logic for robust evaluation
- Pydantic-based result validation

## 🚀 Usage

### Basic Benchmark Run

```bash
cd benchmarker/src
python benchmark-run.py --num-samples 10 --use-async True
```

### Configuration

Edit `config.yml` to customize:

```yaml
agent_name: query-agent          # or any DSPy variant
dataset: freshstack-godot        # dataset to benchmark
reload_database: False          # whether to reload data
batch_size: 5                   # async batch size
max_concurrent: 5               # max concurrent requests
```

### Environment Variables

Required environment variables:
```bash
export WEAVIATE_URL="your-cluster-url"
export WEAVIATE_API_KEY="your-api-key"
export OPENAI_API_KEY="your-openai-key"
```

## 📊 Evaluation Methodology

### 1. Information Retrieval Evaluation
- **Standard Recall**: Traditional relevant documents retrieved / total relevant
- **Nugget Recall**: Percentage of information nuggets with at least one relevant document retrieved
- **Source Tracking**: Complete provenance of retrieved documents

### 2. Answer Quality Evaluation
- **LM-as-a-Judge**: GPT-4 based evaluation with structured reasoning
- **Multiple Inferences**: Configurable number of judge evaluations for reliability
- **Standardized Scoring**: 1-5 scale with detailed criteria

### 3. Performance Metrics
- **Latency**: Per-query execution time
- **Throughput**: Queries processed per second
- **Resource Usage**: Token consumption tracking for LM-based agents
- **Operation Counts**: Search and aggregation operation tracking

## 🔄 Async Architecture

The benchmarker includes sophisticated async support for high-performance evaluation:

**Concurrency Control:**
- Semaphore-based rate limiting
- Configurable batch processing
- Exponential backoff for failures

**Error Handling:**
- Automatic retry with backoff
- Graceful degradation for partial failures
- Comprehensive error logging

**Progress Tracking:**
- Real-time batch completion updates
- Success/failure rate monitoring
- Sample result previews

## 📈 Output and Results

Results are automatically saved with:
- **Structured JSON**: Complete results with metadata
- **Markdown Reports**: Human-readable summaries
- **Metrics Dashboard**: Key performance indicators
- **Error Analysis**: Detailed failure tracking

The system generates comprehensive reports including:
- Average response quality scores
- Retrieval performance metrics
- Latency and throughput analysis
- Error rates and failure modes

## 🧪 Research Applications

This benchmarker is designed for:
- **Agent Comparison**: Head-to-head evaluation of different agent architectures
- **RAG Research**: Systematic evaluation of retrieval strategies
- **Dataset Analysis**: Understanding query complexity and retrieval challenges
- **Performance Optimization**: Identifying bottlenecks and optimization opportunities

