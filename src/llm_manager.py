import torch
import gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- DEĞİŞTİRİLEN KISIM: Ayar Yükleyici Eklendi ---
from src.utils import load_config

class LLMManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        # --- DEĞİŞTİRİLEN KISIM: YAML Ayarları Dinamik Olarak Okunuyor ---
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
        print(f"LLM Manager Başlatıldı. (Cihaz: {self.device})")

    def load_vision_model(self):
        """Qwen-VL Modelini INT4 formatında (4-bit) yükler."""
        if self.vision_model is None:
            # --- DEĞİŞTİRİLEN KISIM: vision_model_path kullanılıyor ---
            print(f"Yükleniyor: {self.vision_model_path} (4-Bit Quantized)")
            try:
                # 4-bit Quantization Ayarları
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16 
                )

                self.vision_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.vision_model_path,
                    device_map="auto",
                    quantization_config=bnb_config,
                    trust_remote_code=True
                )
                
                self.vision_processor = AutoProcessor.from_pretrained(
                    self.vision_model_path,
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
            # --- DEĞİŞTİRİLEN KISIM: embedding_model_path kullanılıyor ---
            print(f" Yükleniyor: {self.embedding_model_path}")
            self.embedder = SentenceTransformer(self.embedding_model_path, device=self.device)
            print(" Embedding Model Hazır.")
        return self.embedder

    def load_reranker(self):
        """Reranker (Sıralayıcı) Modelini yükler."""
        if self.reranker is None:
            # --- DEĞİŞTİRİLEN KISIM: rerank_model_path kullanılıyor ---
            print(f"Yükleniyor: {self.rerank_model_path}")
            self.reranker = CrossEncoder(self.rerank_model_path, device=self.device, trust_remote_code=True)
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