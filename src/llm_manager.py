import logging
import threading
from typing import Any, Optional, Tuple
import requests
import torch
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from src.utils import load_config

logger = logging.getLogger(__name__)


class VisionAPIClient:
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True
    )
    def generate(self, messages: list, max_tokens: int = 2048) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0
        }
        
        endpoint = f"{self.base_url}/v1/chat/completions"
        
        try:
            response = requests.post(endpoint, json=payload, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            
            error_text = response.text
            
            if response.status_code >= 500 or response.status_code == 429:
                logger.warning(f"Sunucu hatasi alindi ({response.status_code}), tekrar deneniyor...")
                response.raise_for_status()
                
            raise RuntimeError(f"API Iletisim Hatasi ({response.status_code}): {error_text}")
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"API baglanti sorunu, tekrar deneniyor... Hata: {e}")
            raise


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
        self.api_base_url = self.model_cfg.get("api_base_url", "http://127.0.0.1:8000")
        
        self.vision_api_model_name = self.model_cfg.get("vision_model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
        
        try:
            self.embedding_model_path = self.model_cfg["embedding_model"]
            self.rerank_model_path = self.model_cfg["reranker_model"]
            self.vision_model_path = self.model_cfg.get("vision_model", "./local_models/Qwen2.5-VL-7B-Instruct")
        except KeyError as e:
            error_msg = f"Ayar dosyasinda zorunlu model yolu eksik: {e}"
            logger.error(error_msg)
            raise KeyError(error_msg)
        
        self.vision_client: Optional[VisionAPIClient] = None
        self.embedder: Optional[Any] = None
        self.reranker: Optional[Any] = None
        
        self.vision_local_model: Optional[Any] = None
        self.vision_processor: Optional[Any] = None
        
        logger.info(f"LLM Manager baslatildi. (Cihaz: {self.device})")

    def get_vision_client(self) -> VisionAPIClient:
        if self.vision_client is None:
            self.vision_client = VisionAPIClient(
                base_url=self.api_base_url,
                model_name=self.vision_api_model_name
            )
            logger.info(f"Vision API Istemcisi hazir. (URL: {self.api_base_url})")
        return self.vision_client

    def load_vision_model(self) -> Tuple[Any, Any]:
        if self.vision_local_model is None or self.vision_processor is None:
            logger.info(f"Yerel Vision modeli yukleniyor: {self.vision_model_path}")
            
            self.vision_processor = AutoProcessor.from_pretrained(self.vision_model_path)
            
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            
            self.vision_local_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.vision_model_path,
                torch_dtype=dtype,
                device_map=self.device
            )
            logger.info("Yerel Vision modeli ve islemcisi hazir.")
            
        return self.vision_local_model, self.vision_processor
        
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