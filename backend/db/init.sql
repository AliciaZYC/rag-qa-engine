-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table with vector embeddings
-- UPDATED: Changed embedding dimension from 384 to 768 for LegalBERT support
-- Falls back to 384 if using MiniLM
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(768),  -- 768 for LegalBERT, 384 for MiniLM (adjustable)
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- NEW COLUMNS for enhanced legal document support
    provision_label TEXT,              -- LEDGAR provision type (e.g., 'indemnification', 'governing_law')
    section_number TEXT,               -- Section identifier (e.g., 'Article 5', 'Section 3.2')
    chunk_method TEXT,                 -- Chunking strategy used (e.g., 'hybrid_structural_semantic')
    token_count INTEGER                -- Approximate token count for the chunk
);

-- Create index for vector similarity search
-- Using HNSW for better accuracy on smaller datasets (<1M documents)
-- The index is created here but can be recreated after data ingestion for optimization
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Alternative for larger datasets (uncomment if needed):
-- DROP INDEX IF EXISTS documents_embedding_idx;
-- CREATE INDEX documents_embedding_idx ON documents 
-- USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create index on metadata for filtering (UNCHANGED)
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING GIN (metadata);

-- NEW: Create indexes on new metadata columns for faster filtering
CREATE INDEX IF NOT EXISTS idx_provision_label ON documents(provision_label);
CREATE INDEX IF NOT EXISTS idx_section_number ON documents(section_number);
CREATE INDEX IF NOT EXISTS idx_chunk_method ON documents(chunk_method);
CREATE INDEX IF NOT EXISTS idx_created_at ON documents(created_at);

-- Optional: Create a query logs table for tracking (UNCHANGED)
CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    response TEXT,
    retrieved_doc_ids INTEGER[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add helpful comments for documentation
COMMENT ON TABLE documents IS 'Legal document chunks with vector embeddings and provision metadata';
COMMENT ON COLUMN documents.embedding IS 'Vector embedding (768-dim for LegalBERT, 384-dim for MiniLM)';
COMMENT ON COLUMN documents.provision_label IS 'LEDGAR provision type extracted from document metadata';
COMMENT ON COLUMN documents.section_number IS 'Section header or identifier (e.g., Article 5, Section 3.2)';
COMMENT ON COLUMN documents.chunk_method IS 'Chunking algorithm used (e.g., hybrid_structural_semantic, fixed_size)';
COMMENT ON COLUMN documents.token_count IS 'Approximate token count (useful for context window management)';