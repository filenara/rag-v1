import base64
import logging
from io import BytesIO
from typing import List
from PIL import Image

from src.llm_manager import LLMManager

logger = logging.getLogger(__name__)


class VisionProcessor:
    def __init__(self):
        self.llm_manager = LLMManager()
        self.client = self.llm_manager.get_vision_client()

    def _encode_image(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    def generate_captions(self, images: List[Image.Image], prompt: str) -> List[str]:
        captions = []
        for img in images:
            try:
                img_url = self._encode_image(img)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": img_url}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                caption = self.client.generate(messages, max_tokens=512)
                captions.append(caption.strip())
            except Exception as e:
                logger.error("Gorsel analizi sirasinda hata: %s", e)
                captions.append("Gorsel analiz edilemedi.")
        return captions