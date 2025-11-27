"""
Data Ingestion Script for RAG QA Engine
Loads parquet data, chunks into 500-character segments, generates embeddings, and stores in PostgreSQL
"""
import os
import sys
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

# Configuration
CHUNK_SIZE = 1000  # characters per chunk
BATCH_SIZE = 100  # number of chunks to insert at once
MAX_CHUNKS = 100  # maximum number of chunks to process (None for all)
DATA_PATH = '/app/data/train.parquet'

class DataIngestion:
    def __init__(self):
        self.conn = None
        self.model = None
        self.connect_db()
        self.load_model()
    
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
    
    def load_model(self):
        """Load sentence transformer model for embeddings"""
        try:
            print("Loading embedding model (this may take a moment)...")
            # Using all-MiniLM-L6-v2: lightweight, fast, 384-dimensional embeddings
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print(f"✓ Model loaded: all-MiniLM-L6-v2 (384 dimensions)")
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            sys.exit(1)
    
    def chunk_text(self, text, chunk_size=CHUNK_SIZE):
        """
        Split text into chunks of approximately chunk_size characters
        Tries to break at sentence boundaries when possible
        """
        if not text or len(text) == 0:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            
            # If we're not at the end, try to break at a sentence boundary
            if end < text_len:
                # Look for sentence endings in the last 50 characters of the chunk
                chunk_text = text[start:end]
                sentence_breaks = ['. ', '! ', '? ', '\n\n']
                best_break = -1
                
                for break_char in sentence_breaks:
                    pos = chunk_text.rfind(break_char)
                    if pos > chunk_size * 0.7:  # Only break if it's reasonably far in
                        best_break = max(best_break, pos + len(break_char))
                
                if best_break > 0:
                    end = start + best_break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end
        
        return chunks
    
    def load_and_chunk_data(self):
        """Load parquet data and create chunks"""
        try:
            print(f"\nLoading data from: {DATA_PATH}")
            df = pd.read_parquet(DATA_PATH)
            print(f"✓ Loaded {len(df)} rows")
            print(f"  Columns: {list(df.columns)}")
            
            # Determine which columns contain text content
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            print(f"  Text columns: {text_columns}")
            
            all_chunks = []
            
            for idx, row in df.iterrows():
                # Check if we've reached the maximum number of chunks
                if MAX_CHUNKS and len(all_chunks) >= MAX_CHUNKS:
                    print(f"  Reached maximum chunk limit ({MAX_CHUNKS}), stopping at row {idx}")
                    break
                
                # Combine all text columns for this row
                text_parts = []
                metadata = {}
                
                for col in df.columns:
                    value = row[col]
                    if pd.notna(value):
                        if col in text_columns:
                            text_parts.append(f"{col}: {value}")
                        metadata[col] = str(value)
                
                # Create full text from all columns
                full_text = " | ".join(text_parts)
                
                # Chunk the text
                chunks = self.chunk_text(full_text)
                
                for chunk_idx, chunk in enumerate(chunks):
                    # Check chunk limit again
                    if MAX_CHUNKS and len(all_chunks) >= MAX_CHUNKS:
                        break
                        
                    chunk_metadata = metadata.copy()
                    chunk_metadata['source_row'] = int(idx)
                    chunk_metadata['chunk_index'] = chunk_idx
                    chunk_metadata['total_chunks'] = len(chunks)
                    
                    all_chunks.append({
                        'content': chunk,
                        'metadata': chunk_metadata
                    })
                
                # Progress indicator
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(df)} rows, {len(all_chunks)} chunks created")
            
            print(f"\n✓ Created {len(all_chunks)} chunks from {len(df)} rows")
            return all_chunks
        
        except Exception as e:
            print(f"✗ Data loading failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def generate_embeddings(self, chunks):
        """Generate embeddings for all chunks"""
        try:
            print(f"\nGenerating embeddings for {len(chunks)} chunks...")
            texts = [chunk['content'] for chunk in chunks]
            
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            print(f"✓ Generated embeddings with shape: {embeddings.shape}")
            return embeddings
        
        except Exception as e:
            print(f"✗ Embedding generation failed: {e}")
            sys.exit(1)
    
    def insert_chunks(self, chunks, embeddings):
        """Insert chunks with embeddings into database"""
        try:
            import json
            
            print(f"\nInserting {len(chunks)} chunks into database...")
            
            cursor = self.conn.cursor()
            
            # Prepare data for batch insert
            data = []
            for chunk, embedding in zip(chunks, embeddings):
                data.append((
                    chunk['content'],
                    embedding.tolist(),
                    json.dumps(chunk['metadata'])
                ))
            
            # Batch insert
            insert_query = """
                INSERT INTO documents (content, embedding, metadata)
                VALUES (%s, %s, %s::jsonb)
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
        """Create vector similarity indexes after data ingestion"""
        try:
            print("\nCreating vector similarity index...")
            cursor = self.conn.cursor()
            
            # Get document count to decide on index type
            cursor.execute("SELECT COUNT(*) FROM documents")
            count = cursor.fetchone()[0]
            
            if count < 100000:
                # HNSW for smaller datasets (better accuracy)
                print(f"  Using HNSW index (document count: {count})")
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                    ON documents USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                """)
            else:
                # IVFFlat for larger datasets (faster build)
                print(f"  Using IVFFlat index (document count: {count})")
                lists = min(count // 1000, 100)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                    ON documents USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {lists});
                """)
            
            self.conn.commit()
            print("✓ Vector index created successfully")
            
            cursor.close()
        
        except Exception as e:
            print(f"✗ Index creation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Main ingestion pipeline"""
        start_time = datetime.now()
        print("\n" + "=" * 60)
        print("RAG DATA INGESTION PIPELINE")
        print("=" * 60)
        
        try:
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
        
        except Exception as e:
            print(f"\n✗ Ingestion failed: {e}")
            sys.exit(1)
        finally:
            if self.conn:
                self.conn.close()

if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run()

