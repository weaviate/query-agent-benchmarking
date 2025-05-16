import weaviate
import json
import pandas as pd
import os
import weaviate.classes as wvc

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Connect to Weaviate
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
    headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
)

def reset_database():
    """Clear all existing collections"""
    client.collections.delete_all()
    print("Database reset completed")

def create_schema(class_name, description, properties):
    """Create a single schema in Weaviate"""
    # Map Python/JSON types to Weaviate types
    type_mapping = {
        "string": wvc.config.DataType.TEXT,
        "number": wvc.config.DataType.NUMBER,
        "boolean": wvc.config.DataType.BOOL,
        "text": wvc.config.DataType.TEXT
    }
    
    # Convert properties to Weaviate format
    weaviate_properties = []
    for prop in properties:
        data_type = type_mapping.get(prop['data_type'][0].lower(), wvc.config.DataType.TEXT)
        weaviate_properties.append(
            wvc.config.Property(
                name=prop['name'],
                description=prop['description'],
                data_type=data_type
            )
        )
    
    # Create collection
    client.collections.create(
        name=class_name,
        description=description,
        properties=weaviate_properties,
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai()
    )
    
    print(f"Created collection: {class_name}")

def populate_data(collection_name, csv_path):
    """Populate a collection with data from a CSV file"""
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Get collection
        collection = client.collections.get(collection_name)
        
        # Convert DataFrame to records
        records = df.to_dict('records')
        
        # Insert records one by one
        for record in records:
            collection.data.insert(properties=record)
        
        print(f"Added {len(records)} records to {collection_name}")
        
    except Exception as e:
        print(f"Error populating data for {collection_name}: {str(e)}")

def process_schemas(schemas_file_path, data_dir):
    """Process schemas file to create collections and populate with data"""
    # Reset database
    reset_database()
    
    # Load schemas
    with open(schemas_file_path, 'r') as f:
        schema_data = json.load(f)
    
    # Track processed schemas
    processed_schemas = set()
    
    # Process each collection in the schema
    for collection in schema_data['weaviate_collections']:
        collection_name = collection['name']
        
        # Skip if already processed
        if collection_name in processed_schemas:
            continue
            
        # Create schema
        create_schema(
            class_name=collection_name,
            description=collection['envisioned_use_case_overview'],
            properties=collection['properties']
        )
        
        # Populate data if CSV file exists
        csv_path = os.path.join(data_dir, f"{collection_name}.csv")
        if os.path.exists(csv_path):
            populate_data(collection_name, csv_path)
        else:
            print(f"No CSV file found for {collection_name}")
        
        processed_schemas.add(collection_name)
    
    print(f"\nProcessed {len(processed_schemas)} unique schemas")
    return list(processed_schemas)

def main():
    # Define paths
    schemas_file_path = "../data/schemas/schemas.json"
    data_dir = "../data/records"
    
    try:
        # Process schemas file and populate database
        processed_schemas = process_schemas(
            schemas_file_path=schemas_file_path,
            data_dir=data_dir
        )
        
        print("\nDatabase setup completed successfully!")
        print("Created and populated the following collections:")
        for schema in processed_schemas:
            print(f"- {schema}")
            
    except Exception as e:
        print(f"\nError during database setup: {str(e)}")
    finally:
        # Close Weaviate client connection
        client.close()

if __name__ == "__main__":
    main()