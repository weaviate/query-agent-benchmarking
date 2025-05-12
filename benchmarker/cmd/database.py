import os
import time
import weaviate
import weaviate.collections.classes.config as wvcc

def load_database(
    weaviate_client,
    dataset_name: str, 
    objects: dict
):
    if dataset_name == "enron":
        if weaviate_client.collections.exists("EnronEmails"):
            weaviate_client.collections.delete("EnronEmails")
        
        enron_emails_collection = weaviate_client.collections.create(
            name="EnronEmails",
            vectorizer_config=wvcc.Configure.Vectorizer.text2vec_weaviate(),
            properties=[
                wvcc.Property(name="email_body", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="dataset_id", data_type=wvcc.DataType.INT),
            ],
        )

        start_time = time.time()
        with weaviate_client.batch.fixed_size(batch_size=100, concurrent_requests=4) as batch:
            for i, email_with_id in enumerate(objects):
                batch.add_object(
                    collection="EnronEmails",
                    properties={
                        "email_body": email_with_id["email_body"],
                        "dataset_id": email_with_id["dataset_id"]
                    }
                )
                if i % 1000 == 999:
                    print(f"Inserted {i + 1} emails into Weaviate... (Time elapsed: {time.time()-start_time:.2f} seconds)")

        end_time = time.time()
        upload_time = end_time - start_time
        print(f"Inserted {i + 1} emails into Weaviate... (Time elapsed: {upload_time:.2f} seconds)")