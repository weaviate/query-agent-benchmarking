from locust import HttpUser, task, between
import json
import random

class SimpleRAGTestUser(HttpUser):
    wait_time = between(1, 10)
    
    def on_start(self):
        self.test_questions = [
            "What are the main features of Wix?",
            "How do I create a website with Wix?",
            "What pricing plans does Wix offer?",
            "How do I customize my Wix website?",
            "What SEO tools are available in Wix?",
            "How do I add e-commerce to my Wix site?",
            "What are Wix Apps and how do I use them?",
            "How do I connect a custom domain to Wix?",
            "What are the mobile editing capabilities?",
            "How does Wix handle website security?"
        ]
    
    @task
    def ask_question(self):
        question = random.choice(self.test_questions)
        
        payload = {
            "question": question
        }
        
        with self.client.post("/query", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "answer" in data and len(data["answer"]) > 10:
                        response.success()
                        print(f"✅ Got answer: {data['answer'][:50]}...")
                    else:
                        response.failure("Answer too short or missing")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")