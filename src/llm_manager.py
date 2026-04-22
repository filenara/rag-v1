import logging
import threading
from typing import Dict, List, Optional, Union
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.utils import load_config

logger = logging.getLogger(__name__)


class VisionClient:
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
        
        response = requests.post(endpoint, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        
        if response.status_code >= 500 or response.status_code == 429:
            logger.warning(
                "Sunucu hatasi alindi (%s), tekrar deneniyor...", 
                response.status_code
            )
            response.raise_for_status()
            
        raise RuntimeError(f"API Iletisim Hatasi ({response.status_code}): {response.text}")


class EmbeddingClient:
    def __init__(self, base_url: str):
        self.endpoint = f"{base_url.rstrip('/')}/embed"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True
    )
    def encode(
        self, texts: Union[str, List[str]], normalize_embeddings: bool = True, **kwargs
    ) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
            
        payload = {
            "inputs": texts,
            "normalize": normalize_embeddings
        }
        
        response = requests.post(self.endpoint, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()


class RerankClient:
    def __init__(self, base_url: str):
        self.endpoint = f"{base_url.rstrip('/')}/rerank"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True
    )
    def predict(self, pairs: List[List[str]], **kwargs) -> List[float]:
        payload = {
            "texts": pairs
        }
        
        response = requests.post(self.endpoint, json=payload, timeout=30)
        response.raise_for_status()
        
        results = response.json()
        return [item["score"] for item in results]


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
            error_msg = "Ayar dosyasinda 'models' anahtari bulunamadi."
            logger.error(error_msg)
            raise KeyError(error_msg)
        
        self.vllm_api_url = self.model_cfg.get("vllm_api_url", "http://127.0.0.1:8000")
        self.vision_model_name = self.model_cfg.get(
            "vision_model_name", "Qwen/Qwen3-VL-4B-Instruct"
        )
        self.tei_embed_url = self.model_cfg.get("tei_embed_url", "http://127.0.0.1:8081")
        self.tei_rerank_url = self.model_cfg.get("tei_rerank_url", "http://127.0.0.1:8082")
        
        self.vision_client: Optional[VisionClient] = None
        self.embedder_client: Optional[EmbeddingClient] = None
        self.reranker_client: Optional[RerankClient] = None
        
        logger.info("LLM Manager (Client Modu) baslatildi.")

    def get_vision_client(self) -> VisionClient:
        if self.vision_client is None:
            self.vision_client = VisionClient(
                base_url=self.vllm_api_url,
                model_name=self.vision_model_name
            )
            logger.info("Vision API Istemcisi hazir. URL: %s", self.vllm_api_url)
        return self.vision_client

    def load_embedder(self) -> EmbeddingClient:
        if self.embedder_client is None:
            self.embedder_client = EmbeddingClient(base_url=self.tei_embed_url)
            logger.info("Embedding API Istemcisi hazir. URL: %s", self.tei_embed_url)
        return self.embedder_client

    def load_reranker(self) -> RerankClient:
        if self.reranker_client is None:
            self.reranker_client = RerankClient(base_url=self.tei_rerank_url)
            logger.info("Reranker API Istemcisi hazir. URL: %s", self.tei_rerank_url)
        return self.reranker_client