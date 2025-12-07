"""
Dual Embedding Model: LegalBERT with MiniLM fallback
"""
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from .models import ModelType, EmbeddingConfig

class DualEmbedder:
    """
    Dual-model embedding strategy:
    1. Primary: LegalBERT (768-dim, legal domain)
    2. Fallback: MiniLM (384-dim, fast inference)
    """
    
    def __init__(
        self, 
        primary_model: ModelType = ModelType.LEGAL_BERT,
        fallback_model: ModelType = ModelType.MINI_LM,
        use_fallback: bool = False
    ):
        """
        Args:
            primary_model: Primary embedding model (LegalBERT)
            fallback_model: Fallback model for development (MiniLM)
            use_fallback: If True, use fallback model instead of primary
        """
        self.primary_config = EmbeddingConfig.get_config(primary_model)
        self.fallback_config = EmbeddingConfig.get_config(fallback_model)
        self.use_fallback = use_fallback
        
        # Load appropriate model
        if use_fallback:
            print(f"  [DualEmbedder] Loading fallback model: {self.fallback_config.model_name}")
            print(f"  Dimension: {self.fallback_config.dimension}")
            self.model = SentenceTransformer(self.fallback_config.model_name)
            self.active_config = self.fallback_config
        else:
            print(f"  [DualEmbedder] Loading primary model: {self.primary_config.model_name}")
            print(f"  Dimension: {self.primary_config.dimension}")
            print(f"  Legal domain optimized: {self.primary_config.is_legal_domain}")
            try:
                self.model = SentenceTransformer(self.primary_config.model_name)
                self.active_config = self.primary_config
            except Exception as e:
                print(f"  ⚠️ Failed to load primary model: {e}")
                print(f"  Falling back to {self.fallback_config.model_name}")
                self.model = SentenceTransformer(self.fallback_config.model_name)
                self.active_config = self.fallback_config
                self.use_fallback = True
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query (no batch processing)
        
        Args:
            query: Query text
            
        Returns:
            1D numpy array of embedding
        """
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        return embedding
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.active_config.dimension
    
    def get_model_name(self) -> str:
        """Get active model name"""
        return self.active_config.model_name
    
    def is_legal_optimized(self) -> bool:
        """Check if using legal-domain optimized model"""
        return self.active_config.is_legal_domain
    
    def get_info(self) -> dict:
        """Get model information"""
        return {
            'model_name': self.active_config.model_name,
            'dimension': self.active_config.dimension,
            'max_seq_length': self.active_config.max_seq_length,
            'legal_optimized': self.active_config.is_legal_domain,
            'is_fallback': self.use_fallback
        }