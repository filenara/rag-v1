import logging
import threading
from typing import Any, Tuple, Optional
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.utils import load_config

logger = logging.getLogger(__name__)


class LLMManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> "LLMManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LLMManager, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        self.config = load_config()
        self.model_cfg = self.config.get("models")
        
        if not self.model_cfg:
            error_msg = "Ayar dosyasinda (settings.yaml) 'models' anahtari bulunamadi."
            logger.error(error_msg)
            raise KeyError(error_msg)
        
        self.device = self.model_cfg.get("device", "cpu")
        
        try:
            self.vision_model_path = self.model_cfg["vision_model"]
            self.embedding_model_path = self.model_cfg["embedding_model"]
            self.rerank_model_path = self.model_cfg["reranker_model"]
        except KeyError as e:
            error_msg = f"Ayar dosyasinda zorunlu model yolu eksik: {e}"
            logger.error(error_msg)
            raise KeyError(error_msg)
        
        self.vision_model: Optional[Any] = None
        self.vision_processor: Optional[Any] = None
        self.embedder: Optional[Any] = None
        self.reranker: Optional[Any] = None
        
        logger.info(f"LLM Manager baslatildi. (Cihaz: {self.device})")

    def load_vision_model(self) -> Tuple[Any, Any]:
        if self.vision_model is None:
            logger.info(
                f"Yukleniyor: {self.vision_model_path} (4-Bit Quantized)"
            )
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
                    min_pixels=256 * 28 * 28,
                    max_pixels=1280 * 28 * 28
                )
                logger.info("Vision Model hazir (Optimize edildi).")
            except Exception as e:
                logger.error(f"Vision Model yukleme hatasi: {e}")
                raise e
        else:
            logger.debug("Vision Model zaten hafizada, tekrar yuklenmiyor.")
            
        return self.vision_model, self.vision_processor
        
    def load_embedder(self) -> Any:
        if self.embedder is None:
            logger.info(f"Yukleniyor: {self.embedding_model_path}")
            self.embedder = SentenceTransformer(
                self.embedding_model_path, device=self.device
            )
            logger.info("Embedding Model hazir.")
        return self.embedder

    def load_reranker(self) -> Any:
        if self.reranker is None:
            logger.info(f"Yukleniyor: {self.rerank_model_path}")
            self.reranker = CrossEncoder(
                self.rerank_model_path, device=self.device, trust_remote_code=True
            )
            logger.info("Reranker Model hazir.")
        return self.reranker

    def unload_vision_model(self) -> None:
        if self.vision_model is not None:
            logger.info("Vision Model nesneleri referanslardan siliniyor...")
            del self.vision_model
            del self.vision_processor
            self.vision_model = None
            self.vision_processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Vision Model referanslari temizlendi.")