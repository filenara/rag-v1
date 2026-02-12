from dataclasses import dataclass
import os
import torch

@dataclass
class Config:
    # Model Ayarları
    MODEL_ID: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    EMBED_MODEL: str = "BAAI/bge-m3"
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    
    # Chunking
    CHUNK_SIZE: int = 2000
    CHUNK_OVERLAP: int = 400
    
    # Retrieval
    INITIAL_TOP_K: int = 15
    FINAL_TOP_K: int = 3
    VECTOR_SPAM_LIMIT: int = 600
    
    # Görsel İşleme
    IMAGE_DPI: int = 150
    MIN_IMAGE_DIM: int = 200
    
    # Donanım (Otomatik Algılama)
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
cfg = Config()