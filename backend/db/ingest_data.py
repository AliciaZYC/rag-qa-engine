"""
Upgraded Data Ingestion Pipeline v2.0
Features:
- Hybrid structural-semantic chunking
- LegalBERT embeddings with MiniLM fallback
- Enhanced metadata storage
"""
import os
import sys
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from pgvector.psycopg2 import register_vector
import numpy as np
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunking import HybridChunker
from embeddings import DualEmbedder, ModelType

# Configuration
BATCH_SIZE = 100
MAX_CHUNKS = 100  # Set to None for all data
DATA_PATH = '/app/data/train.parquet'

# Chunking config
CHUNK_MAX_TOKENS = 400
OVERLAP_SENTENCES = 1
SEMANTIC_THRESHOLD = 0.6
USE_SEMANTIC_CHUNKING = True

# Embedding config
USE_LEGAL_BERT = True  # Set to False to use MiniLM fallback

class DataIngestionV2:
    def __init__(self):
        self.conn = None
        self.chunker = None
        self.embedder = None
        self.connect_db()
        self.initialize_models()
    
    def connect_db(self):
        """Connect to PostgreSQL database"""
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
    
    def initialize_models(self):
        """Initialize chunker and embedder"""
        try:
            print("\n" + "=" * 60)
            print("INITIALIZING MODELS")
            print("=" * 60)
            
            # Initialize chunker
            print("\n[1/2] Initializing Hybrid Chunker...")
            self.chunker = HybridChunker(
                max_tokens=CHUNK_MAX_TOKENS,
                overlap_sentences=OVERLAP_SENTENCES,
                semantic_threshold=SEMANTIC_THRESHOLD,
                use_semantic=USE_SEMANTIC_CHUNKING
            )
            print("✓ Chunker initialized")
            
            # Initialize embedder
            print("\n[2/2] Initializing Dual Embedder...")
            self.embedder = DualEmbedder(
                primary_model=ModelType.LEGAL_BERT,
                fallback_model=ModelType.MINI_LM,
                use_fallback=not USE_LEGAL_BERT
            )
            
            embedder_info = self.embedder.get_info()
            print("✓ Embedder initialized")
            print(f"  Active model: {embedder_info['model_name']}")
            print(f"  Dimensions: {embedder_info['dimension']}")
            print(f"  Legal optimized: {embedder_info['legal_optimized']}")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"✗ Model initialization failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def update_schema(self):
        """Update database schema for new metadata columns"""
        try:
            print("\nUpdating database schema...")
            cursor = self.conn.cursor()
            
            # Add new metadata columns if they don't exist
            metadata_columns = [
                ('provision_label', 'VARCHAR(100)'),
                ('section_number', 'VARCHAR(50)'),
                ('chunk_method', 'VARCHAR(50)'),
                ('token_count', 'INTEGER')
            ]
            
            for col_name, col_type in metadata_columns:
                try:
                    cursor.execute(f"""
                        ALTER TABLE documents 
                        ADD COLUMN IF NOT EXISTS {col_name} {col_type}
                    """)
                    print(f"  ✓ Added column: {col_name}")
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        print(f"  ⚠️ Column {col_name}: {e}")
            
            # Update embedding dimension if using LegalBERT
            target_dim = self.embedder.get_dimension()
            if target_dim == 768:
                print(f"\n  Updating embedding dimension to {target_dim}...")
                try:
                    cursor.execute(f"""
                        ALTER TABLE documents 
                        ALTER COLUMN embedding TYPE vector({target_dim})
                    """)
                    print(f"  ✓ Embedding dimension updated to {target_dim}")
                except Exception as e:
                    if "type vector does not exist" in str(e).lower():
                        print(f"  ⚠️ Dimension update skipped: {e}")
                    else:
                        print(f"  ⚠️ Dimension update: {e}")
            
            self.conn.commit()
            print("✓ Schema update complete\n")
            cursor.close()
            
        except Exception as e:
            print(f"✗ Schema update failed: {e}")
            self.conn.rollback()
    
    def load_and_chunk_data(self):
        """Load parquet data and apply hybrid chunking"""
        try:
            print(f"\nLoading data from: {DATA_PATH}")
            df = pd.read_parquet(DATA_PATH)
            print(f"✓ Loaded {len(df)} rows")
            print(f"  Columns: {list(df.columns)}")
            
            # Determine text columns
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            print(f"  Text columns: {text_columns}")
            
            all_chunks = []
            provision_stats = {}
            
            for idx, row in df.iterrows():
                if MAX_CHUNKS and len(all_chunks) >= MAX_CHUNKS:
                    print(f"  Reached maximum chunk limit ({MAX_CHUNKS})")
                    break
                
                # Combine text columns
                text_parts = []
                metadata = {}
                
                for col in df.columns:
                    value = row[col]
                    if pd.notna(value):
                        if col in text_columns:
                            text_parts.append(f"{col}: {value}")
                        metadata[col] = str(value)
                
                full_text = " | ".join(text_parts)
                
                # Apply hybrid chunking
                chunks = self.chunker.chunk_document(full_text, metadata)
                
                for chunk_idx, chunk in enumerate(chunks):
                    if MAX_CHUNKS and len(all_chunks) >= MAX_CHUNKS:
                        break
                    
                    # Track provision distribution
                    label = chunk['provision_label']
                    provision_stats[label] = provision_stats.get(label, 0) + 1
                    
                    # Add source metadata
                    chunk['source_row'] = int(idx)
                    chunk['chunk_index'] = chunk_idx
                    chunk['original_metadata'] = metadata
                    
                    all_chunks.append(chunk)
                
                if (idx + 1) % 50 == 0:
                    print(f"  Processed {idx + 1}/{len(df)} rows, {len(all_chunks)} chunks")
            
            print(f"\n✓ Created {len(all_chunks)} chunks from {idx + 1} rows")
            
            # Display chunking statistics
            stats = self.chunker.get_stats(all_chunks)
            print("\nChunking Statistics:")
            print(f"  Total chunks: {stats['total_chunks']}")
            print(f"  Avg tokens/chunk: {stats['avg_tokens']:.1f}")
            print(f"  Token range: [{stats['min_tokens']}, {stats['max_tokens']}]")
            print(f"\n  Provision distribution:")
            for label, count in sorted(provision_stats.items(), key=lambda x: -x[1]):
                print(f"    {label}: {count} chunks")
            
            return all_chunks
        
        except Exception as e:
            print(f"✗ Data loading failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def generate_embeddings(self, chunks):
        """Generate embeddings using dual embedder"""
        try:
            print(f"\nGenerating embeddings for {len(chunks)} chunks...")
            texts = [chunk['text'] for chunk in chunks]
            
            embeddings = self.embedder.encode(
                texts,
                batch_size=32,
                show_progress=True
            )
            
            print(f"✓ Generated embeddings: {embeddings.shape}")
            return embeddings
        
        except Exception as e:
            print(f"✗ Embedding generation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def insert_chunks(self, chunks, embeddings):
        """Insert chunks with enhanced metadata"""
        try:
            print(f"\nInserting {len(chunks)} chunks into database...")
            cursor = self.conn.cursor()
            
            # Prepare batch data
            data = []
            for chunk, embedding in zip(chunks, embeddings):
                data.append((
                    chunk['text'],
                    embedding.tolist(),
                    json.dumps(chunk['original_metadata']),
                    chunk.get('provision_label'),
                    chunk.get('section_number'),
                    chunk.get('chunk_method'),
                    chunk.get('token_count')
                ))
            
            # Batch insert
            insert_query = """
                INSERT INTO documents 
                (content, embedding, metadata, provision_label, section_number, chunk_method, token_count)
                VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s)
            """
            
            execute_batch(cursor, insert_query, data, page_size=BATCH_SIZE)
            self.conn.commit()
            
            # Verify insertion
            cursor.execute("SELECT COUNT(*) FROM documents")
            count = cursor.fetchone()[0]
            
            print(f"✓ Successfully inserted {len(chunks)} chunks")
            print(f"✓ Total documents in database: {count}")
            
            cursor.close()
        
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Database insertion failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def create_indexes(self):
        """Create optimized vector indexes"""
        try:
            print("\nCreating vector similarity index...")
            cursor = self.conn.cursor()
            
            # Drop existing index if it exists
            cursor.execute("DROP INDEX IF EXISTS documents_embedding_idx")
            
            # Get document count
            cursor.execute("SELECT COUNT(*) FROM documents")
            count = cursor.fetchone()[0]
            
            # Create HNSW index (optimal for <1M documents)
            print(f"  Creating HNSW index (document count: {count})")
            cursor.execute("""
                CREATE INDEX documents_embedding_idx 
                ON documents USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
            
            # Create indexes on metadata columns for filtering
            print("  Creating metadata indexes...")
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_provision_label 
                ON documents(provision_label);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_section_number 
                ON documents(section_number);
            """)
            
            self.conn.commit()
            print("✓ All indexes created successfully")
            cursor.close()
        
        except Exception as e:
            print(f"✗ Index creation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Main ingestion pipeline"""
        start_time = datetime.now()
        print("\n" + "=" * 60)
        print("RAG DATA INGESTION PIPELINE V2.0")
        print("Hybrid Chunking + LegalBERT Embeddings")
        print("=" * 60)
        
        try:
            # Step 0: Update schema
            self.update_schema()
            
            # Step 1: Load and chunk data
            chunks = self.load_and_chunk_data()
            
            # Step 2: Generate embeddings
            embeddings = self.generate_embeddings(chunks)
            
            # Step 3: Insert into database
            self.insert_chunks(chunks, embeddings)
            
            # Step 4: Create indexes
            self.create_indexes()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            print("\n" + "=" * 60)
            print(f"✓ INGESTION COMPLETE in {elapsed:.2f}s")
            print("=" * 60)
            
            # Display summary
            print("\nSummary:")
            print(f"  Chunks processed: {len(chunks)}")
            print(f"  Embedding model: {self.embedder.get_model_name()}")
            print(f"  Embedding dimension: {self.embedder.get_dimension()}")
            print(f"  Chunking method: Hybrid structural-semantic")
            
        except Exception as e:
            print(f"\n✗ Ingestion failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            if self.conn:
                self.conn.close()

if __name__ == "__main__":
    ingestion = DataIngestionV2()
    ingestion.run()