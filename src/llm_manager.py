import torch
import gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
from config.settings import models

class LLMManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.device = models.device
        self.vision_model = None
        self.vision_processor = None
        self.embedder = None
        self.reranker = None
        print(f"LLM Manager Başlatıldı. (Cihaz: {self.device})")

    def load_vision_model(self):
        """Qwen-VL Modelini INT4 formatında (4-bit) yükler."""
        if self.vision_model is None:
            print(f"Yükleniyor: {models.vision_model} (4-Bit Quantized)")
            try:
                # 4-bit Quantization Ayarları
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    # P100 için torch.float16 daha stabil olabilir, yeni nesil kartlar için bfloat16
                    bnb_4bit_compute_dtype=torch.float16 
                )

                self.vision_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    models.vision_model,
                    device_map="auto",
                    quantization_config=bnb_config,
                    trust_remote_code=True
                )
                
                self.vision_processor = AutoProcessor.from_pretrained(
                    models.vision_model,
                    trust_remote_code=True,
                    min_pixels=256*28*28,
                    max_pixels=1280*28*28
                )
                print(" Vision Model Hazır (Optimize Edildi).")
            except Exception as e:
                print(f" Vision Model Hatası: {e}")
                raise e
        else:
            print("Vision Model zaten hafızada, tekrar yüklenmiyor.")
            
        return self.vision_model, self.vision_processor
        
    def load_embedder(self):
        """Embedding (Vektör) Modelini yükler."""
        if self.embedder is None:
            print(f" Yükleniyor: {models.embedding_model}")
            self.embedder = SentenceTransformer(models.embedding_model, device=self.device)
            print(" Embedding Model Hazır.")
        return self.embedder

    def load_reranker(self):
        """Reranker (Sıralayıcı) Modelini yükler."""
        if self.reranker is None:
            print(f"Yükleniyor: {models.rerank_model}")
            self.reranker = CrossEncoder(models.rerank_model, device=self.device, trust_remote_code=True)
            print(" Reranker Model Hazır.")
        return self.reranker

    def unload_vision_model(self):
        """
        Ağır Vision modelini hafızadan siler. 
        Ingestion bittiğinde veya sadece metin araması yaparken çağrılabilir.
        """
        if self.vision_model is not None:
            print(" Vision Model hafızadan temizleniyor...")
            del self.vision_model
            del self.vision_processor
            self.vision_model = None
            self.vision_processor = None
            gc.collect()
            torch.cuda.empty_cache()
            print(" Hafıza Temizlendi.")