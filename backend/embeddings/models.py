"""
Embedding model configurations
"""
from enum import Enum
from dataclasses import dataclass

class ModelType(Enum):
    """Available embedding models"""
    LEGAL_BERT = "nlpaueb/legal-bert-base-uncased"
    MINI_LM = "all-MiniLM-L6-v2"
    BGE_BASE = "BAAI/bge-base-en-v1.5"

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""
    model_name: str
    dimension: int
    max_seq_length: int
    is_legal_domain: bool
    
    @staticmethod
    def get_config(model_type: ModelType):
        """Get configuration for a specific model type"""
        configs = {
            ModelType.LEGAL_BERT: EmbeddingConfig(
                model_name=model_type.value,
                dimension=768,
                max_seq_length=512,
                is_legal_domain=True
            ),
            ModelType.MINI_LM: EmbeddingConfig(
                model_name=model_type.value,
                dimension=384,
                max_seq_length=512,
                is_legal_domain=False
            ),
            ModelType.BGE_BASE: EmbeddingConfig(
                model_name=model_type.value,
                dimension=768,
                max_seq_length=512,
                is_legal_domain=False
            )
        }
        return configs[model_type]