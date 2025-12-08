"""
Evaluation Script for RAG Retrieval Quality
Compares retrieval performance with test queries
"""
import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings import DualEmbedder, ModelType

# Test queries for legal documents
TEST_QUERIES = [
    {
        'query': 'What are the indemnification provisions?',
        'expected_labels': ['indemnification'],
        'description': 'Legal liability clauses'
    },
    {
        'query': 'Which law governs this agreement?',
        'expected_labels': ['governing_law'],
        'description': 'Jurisdiction and applicable law'
    },
    {
        'query': 'What are the confidentiality requirements?',
        'expected_labels': ['confidentiality'],
        'description': 'Non-disclosure provisions'
    },
    {
        'query': 'How can this contract be terminated?',
        'expected_labels': ['termination'],
        'description': 'Termination clauses'
    },
    {
        'query': 'What happens in case of force majeure?',
        'expected_labels': ['force_majeure'],
        'description': 'Unforeseeable circumstances'
    }
]

class RetrievalEvaluator:
    """Evaluate retrieval quality of RAG system"""
    
    def __init__(self, use_legal_bert=True):
        """
        Args:
            use_legal_bert: If True, use LegalBERT; if False, use MiniLM
        """
        self.conn = None
        self.embedder = None
        self.use_legal_bert = use_legal_bert
        self.connect_db()
        self.load_embedder()
    
    def connect_db(self):
        """Connect to database"""
        try:
            self.conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'postgres'),
                port=os.getenv('DB_PORT', '5432'),
                database=os.getenv('DB_NAME', 'rag_db'),
                user=os.getenv('DB_USER', 'rag_user'),
                password=os.getenv('DB_PASSWORD', 'rag_password')
            )
            register_vector(self.conn)
            print("✓ Database connected")
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            sys.exit(1)
    
    def load_embedder(self):
        """Load embedding model"""
        model_name = "LegalBERT" if self.use_legal_bert else "MiniLM"
        print(f"Loading {model_name} embedder...")
        self.embedder = DualEmbedder(
            primary_model=ModelType.LEGAL_BERT,
            use_fallback=not self.use_legal_bert
        )
        print(f"✓ Loaded: {self.embedder.get_model_name()}")
    
    def retrieve_top_k(self, query: str, k: int = 5):
        """
        Retrieve top-k most similar documents
        
        Returns:
            List of document dictionaries
        """
        # Generate query embedding
        query_embedding = self.embedder.encode_query(query)
        
        # Vector similarity search
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT 
                id,
                content,
                provision_label,
                section_number,
                chunk_method,
                1 - (embedding <=> %s::vector) as similarity
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding.tolist(), query_embedding.tolist(), k))
        
        results = cursor.fetchall()
        cursor.close()
        
        return results
    
    def calculate_metrics(self, retrieved_labels, expected_labels):
        """
        Calculate Precision@k and Recall@k
        
        Args:
            retrieved_labels: List of retrieved provision labels
            expected_labels: List of expected provision labels
            
        Returns:
            Dict with precision and recall
        """
        retrieved_set = set(retrieved_labels)
        expected_set = set(expected_labels)
        
        if not expected_set:
            return {'precision': 0.0, 'recall': 0.0, 'hits': 0}
        
        hits = len(retrieved_set & expected_set)
        precision = hits / len(retrieved_labels) if retrieved_labels else 0.0
        recall = hits / len(expected_set)
        
        return {
            'precision': precision,
            'recall': recall,
            'hits': hits
        }
    
    def evaluate(self, k=5):
        """
        Evaluate retrieval quality on test queries
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Evaluation results dictionary
        """
        print("\n" + "=" * 60)
        print(f"EVALUATION: {'LegalBERT' if self.use_legal_bert else 'MiniLM'}")
        print("=" * 60)
        
        all_precisions = []
        all_recalls = []
        
        for i, test_case in enumerate(TEST_QUERIES, 1):
            query = test_case['query']
            expected_labels = test_case['expected_labels']
            description = test_case['description']
            
            print(f"\n[Query {i}/{len(TEST_QUERIES)}] {description}")
            print(f"Query: \"{query}\"")
            print(f"Expected labels: {expected_labels}")
            
            # Retrieve documents
            results = self.retrieve_top_k(query, k=k)
            
            # Extract retrieved labels
            retrieved_labels = [r['provision_label'] for r in results if r['provision_label']]
            
            # Calculate metrics
            metrics = self.calculate_metrics(retrieved_labels, expected_labels)
            
            all_precisions.append(metrics['precision'])
            all_recalls.append(metrics['recall'])
            
            print(f"Retrieved labels: {retrieved_labels}")
            print(f"Precision@{k}: {metrics['precision']:.3f}")
            print(f"Recall@{k}: {metrics['recall']:.3f}")
            print(f"Hits: {metrics['hits']}/{len(expected_labels)}")
            
            # Show top result
            if results:
                top_result = results[0]
                print(f"\nTop result (similarity: {top_result['similarity']:.3f}):")
                print(f"  Label: {top_result['provision_label']}")
                print(f"  Content preview: {top_result['content'][:150]}...")
        
        # Overall metrics
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        
        print("\n" + "=" * 60)
        print("OVERALL METRICS")
        print("=" * 60)
        print(f"Average Precision@{k}: {avg_precision:.3f}")
        print(f"Average Recall@{k}: {avg_recall:.3f}")
        
        return {
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'individual_results': list(zip(all_precisions, all_recalls))
        }
    
    def inspect_database(self):
        """Show database statistics"""
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
        # Total documents
        cursor.execute("SELECT COUNT(*) as total FROM documents")
        total = cursor.fetchone()['total']
        
        # Provision distribution
        cursor.execute("""
            SELECT provision_label, COUNT(*) as count
            FROM documents
            WHERE provision_label IS NOT NULL
            GROUP BY provision_label
            ORDER BY count DESC
        """)
        provisions = cursor.fetchall()
        
        # Chunking methods
        cursor.execute("""
            SELECT chunk_method, COUNT(*) as count
            FROM documents
            WHERE chunk_method IS NOT NULL
            GROUP BY chunk_method
        """)
        methods = cursor.fetchall()
        
        print("\n" + "=" * 60)
        print("DATABASE STATISTICS")
        print("=" * 60)
        print(f"Total documents: {total}")
        
        if provisions:
            print(f"\nProvision distribution:")
            for p in provisions[:10]:  # Top 10
                print(f"  {p['provision_label']}: {p['count']}")
        
        if methods:
            print(f"\nChunking methods:")
            for m in methods:
                print(f"  {m['chunk_method']}: {m['count']}")
        
        cursor.close()

if __name__ == "__main__":
    evaluator = RetrievalEvaluator(use_legal_bert=True)
    
    # Inspect database first
    evaluator.inspect_database()
    
    # Run evaluation
    evaluator.evaluate(k=5)