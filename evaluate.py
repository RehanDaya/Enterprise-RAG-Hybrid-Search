# evaluate.py
import json
from app.retriever.vector_store import VectorStore
from app.retriever.enhanced_retriever import EnhancedRetriever
from app.agent.knowledge_agent import KnowledgeAgent
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

class Evaluator:
    def __init__(self, test_questions_path):
        # Load test questions
        with open(test_questions_path, 'r') as f:
            self.test_questions = json.load(f)
        
        # Initialize system
        store = VectorStore()
        db = store.load_existing()
        self.retriever = EnhancedRetriever(db)
        self.agent = KnowledgeAgent(self.retriever)
        
        # Metrics
        self.results = []
    
    def evaluate_retrieval(self):
        for item in self.test_questions:
            query = item["question"]
            expected_docs = item.get("relevant_docs", [])
            
            # Get retrieval results
            results = self.retriever.retrieve(query)
            retrieved_docs = [doc.metadata.get('source') for doc in results['documents']]
            
            # Calculate metrics
            relevant_retrieved = [doc for doc in retrieved_docs if doc in expected_docs]
            precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0
            recall = len(relevant_retrieved) / len(expected_docs) if expected_docs else 1
            
            self.results.append({
                "question": query,
                "precision": precision,
                "recall": recall,
                "retrieved_docs": retrieved_docs,
                "expected_docs": expected_docs
            })
    
    def generate_report(self):
        df = pd.DataFrame(self.results)
        avg_precision = df['precision'].mean()
        avg_recall = df['recall'].mean()
        
        print(f"Evaluation Results:")
        print(f"Average Precision: {avg_precision:.2f}")
        print(f"Average Recall: {avg_recall:.2f}")
        
        # Save detailed results
        df.to_csv("evaluation_results.csv", index=False)
        
        return {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "detailed_results": self.results
        }

# Run evaluation
evaluator = Evaluator("evaluation/test_questions.json")
evaluator.evaluate_retrieval()
report = evaluator.generate_report()