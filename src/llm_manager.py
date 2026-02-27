import torch
import gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.utils import load_config


class LLMManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.config = load_config()
        self.model_cfg = self.config.get('models', {})
        
        self.device = self.model_cfg.get('device', 'cpu')
        self.vision_model_path = self.model_cfg.get('vision_model', 'Qwen/Qwen2.5-VL-7B-Instruct')
        self.embedding_model_path = self.model_cfg.get('embedding_model', 'BAAI/bge-m3')
        self.rerank_model_path = self.model_cfg.get('reranker_model', 'BAAI/bge-reranker-v2-m3')
        
        self.vision_model = None
        self.vision_processor = None
        self.embedder = None
        self.reranker = None
        print(f"LLM Manager Baslatildi. (Cihaz: {self.device})")

    def load_vision_model(self):
        """Qwen-VL Modelini INT4 formatinda (4-bit) yukler."""
        if self.vision_model is None:
            print(f"Yukleniyor: {self.vision_model_path} (4-Bit Quantized)")
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16 
                )

                self.vision_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.vision_model_path,
                    device_map={"": self.device},
                    quantization_config=bnb_config,
                    trust_remote_code=True
                )
                
                self.vision_processor = AutoProcessor.from_pretrained(
                    self.vision_model_path,
                    trust_remote_code=True,
                    min_pixels=256*28*28,
                    max_pixels=1280*28*28
                )
                print("Vision Model Hazir (Optimize Edildi).")
            except Exception as e:
                print(f"Vision Model Hatasi: {e}")
                raise e
        else:
            print("Vision Model zaten hafizada, tekrar yuklenmiyor.")
            
        return self.vision_model, self.vision_processor
        
    def load_embedder(self):
        """Embedding (Vektor) Modelini yukler."""
        if self.embedder is None:
            print(f"Yukleniyor: {self.embedding_model_path}")
            self.embedder = SentenceTransformer(self.embedding_model_path, device=self.device)
            print("Embedding Model Hazir.")
        return self.embedder

    def load_reranker(self):
        """Reranker (Siralayici) Modelini yukler."""
        if self.reranker is None:
            print(f"Yukleniyor: {self.rerank_model_path}")
            self.reranker = CrossEncoder(self.rerank_model_path, device=self.device, trust_remote_code=True)
            print("Reranker Model Hazir.")
        return self.reranker

    def unload_vision_model(self):
        """
        Agir Vision modelini hafizadan siler. 
        Ingestion bittiginde veya sadece metin aramasi yaparken cagrilabilir.
        """
        if self.vision_model is not None:
            print("Vision Model hafizadan temizleniyor...")
            del self.vision_model
            del self.vision_processor
            self.vision_model = None
            self.vision_processor = None
            gc.collect()
            torch.cuda.empty_cache()
            print("Hafiza Temizlendi.")