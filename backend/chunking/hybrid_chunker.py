"""
Hybrid Structural-Semantic Chunker for Legal Documents
Leverages LEDGAR provision labels + semantic chunking
"""
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from .utils import (
    token_count, 
    split_by_sentences, 
    detect_section_header,
    extract_provision_label,
    normalize_legal_text
)

class HybridChunker:
    """
    Hybrid chunker combining:
    1. Structural boundaries (LEDGAR provision labels)
    2. Semantic coherence (sentence similarity)
    3. Size constraints (max tokens)
    """
    
    def __init__(
        self, 
        max_tokens: int = 400,
        overlap_sentences: int = 1,
        semantic_threshold: float = 0.6,
        use_semantic: bool = True
    ):
        """
        Args:
            max_tokens: Maximum tokens per chunk
            overlap_sentences: Number of sentences to overlap
            semantic_threshold: Cosine similarity threshold for semantic chunking
            use_semantic: Whether to use semantic chunking (disable for speed)
        """
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self.semantic_threshold = semantic_threshold
        self.use_semantic = use_semantic
        
        # Load embedding model for semantic chunking (lightweight MiniLM)
        if self.use_semantic:
            print("  [HybridChunker] Loading MiniLM for semantic chunking...")
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.semantic_model = None
    
    def chunk_document(
        self, 
        text: str, 
        metadata: Dict = None
    ) -> List[Dict]:
        """
        Main chunking pipeline
        
        Args:
            text: Document text
            metadata: LEDGAR metadata (contains provision labels)
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Normalize text
        text = normalize_legal_text(text)
        
        # Extract provision label from metadata
        provision_label = extract_provision_label(metadata) if metadata else 'general'
        
        # Step 1: Split by structural boundaries (if detectable)
        provisions = self._split_by_structure(text)
        
        all_chunks = []
        
        for provision in provisions:
            provision_text = provision['text']
            
            # Step 2: Apply semantic chunking within provision
            if self.use_semantic and token_count(provision_text) > self.max_tokens:
                semantic_chunks = self._semantic_chunking(provision_text)
            else:
                semantic_chunks = [provision_text]
            
            # Step 3: Enforce size constraints
            for chunk_text in semantic_chunks:
                if token_count(chunk_text) > self.max_tokens:
                    # Further split if still too large
                    sub_chunks = split_by_sentences(
                        chunk_text, 
                        self.max_tokens, 
                        self.overlap_sentences
                    )
                    for sub_chunk in sub_chunks:
                        all_chunks.append({
                            'text': sub_chunk,
                            'provision_label': provision_label,
                            'section_number': provision.get('section_number'),
                            'chunk_method': 'hybrid_structural_semantic',
                            'token_count': token_count(sub_chunk)
                        })
                else:
                    all_chunks.append({
                        'text': chunk_text,
                        'provision_label': provision_label,
                        'section_number': provision.get('section_number'),
                        'chunk_method': 'hybrid_structural_semantic',
                        'token_count': token_count(chunk_text)
                    })
        
        return all_chunks
    
    def _split_by_structure(self, text: str) -> List[Dict]:
        """
        Split text by structural markers (section headers)
        
        Returns:
            List of provisions with text and section numbers
        """
        lines = text.split('\n')
        provisions = []
        current_provision = {
            'text': '',
            'section_number': None
        }
        
        for line in lines:
            if detect_section_header(line):
                # Save previous provision if it has content
                if current_provision['text'].strip():
                    provisions.append(current_provision)
                
                # Start new provision
                current_provision = {
                    'text': line + '\n',
                    'section_number': line.strip()
                }
            else:
                current_provision['text'] += line + '\n'
        
        # Add last provision
        if current_provision['text'].strip():
            provisions.append(current_provision)
        
        # If no structure detected, return entire text as one provision
        if not provisions:
            provisions = [{
                'text': text,
                'section_number': None
            }]
        
        return provisions
    
    def _semantic_chunking(self, text: str) -> List[str]:
        """
        Split text based on semantic coherence
        
        Uses sentence embeddings to detect topic shifts
        """
        sentences = sent_tokenize(text)
        
        if len(sentences) <= 2:
            return [text]
        
        # Generate sentence embeddings
        embeddings = self.semantic_model.encode(sentences, convert_to_numpy=True)
        
        # Calculate cosine similarities between consecutive sentences
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Cosine similarity between consecutive sentences
            similarity = np.dot(embeddings[i-1], embeddings[i]) / (
                np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
            )
            
            # Check if we should start a new chunk
            should_split = (
                similarity < self.semantic_threshold or 
                token_count(' '.join(current_chunk + [sentences[i]])) > self.max_tokens
            )
            
            if should_split:
                chunks.append(' '.join(current_chunk))
                
                # Add overlap
                if self.overlap_sentences > 0 and len(current_chunk) > self.overlap_sentences:
                    current_chunk = current_chunk[-self.overlap_sentences:] + [sentences[i]]
                else:
                    current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def get_stats(self, chunks: List[Dict]) -> Dict:
        """
        Get statistics about chunked data
        """
        if not chunks:
            return {}
        
        token_counts = [chunk['token_count'] for chunk in chunks]
        provision_counts = {}
        
        for chunk in chunks:
            label = chunk.get('provision_label', 'unknown')
            provision_counts[label] = provision_counts.get(label, 0) + 1
        
        return {
            'total_chunks': len(chunks),
            'avg_tokens': np.mean(token_counts),
            'min_tokens': np.min(token_counts),
            'max_tokens': np.max(token_counts),
            'provision_distribution': provision_counts
        }