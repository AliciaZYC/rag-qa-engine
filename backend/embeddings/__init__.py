"""
Embedding models for legal document retrieval
"""
from .dual_embedder import DualEmbedder
from .models import EmbeddingConfig, ModelType

__all__ = ['DualEmbedder', 'EmbeddingConfig', 'ModelType']