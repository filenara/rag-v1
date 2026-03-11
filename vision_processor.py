%%writefile vision_processor.py
import logging
import re
import torch
from typing import List
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from src.utils import load_config

logger = logging.getLogger(__name__)

class VisionProcessor:
    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens
        self.config = load_config()
        self.batch_size = self.config.get("models", {}).get("vision_batch_size", 2)
        
        vision_model_path = self.config.get("models", {}).get("vision_model", "Qwen/Qwen2.5-VL-7B-Instruct")
        device_map = self.config.get("models", {}).get("device", "cuda")
        
        logger.info(f"Vision modeli yukleniyor: {vision_model_path}")
        
        self.processor = AutoProcessor.from_pretrained(vision_model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vision_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map=device_map
        )

    def generate_captions(self, images: List[Image.Image], prompt_text: str) -> List[str]:
        if not images:
            return []

        batch_size = self.batch_size
        all_captions = []
        prompts = [prompt_text] * len(images)

        for b_idx in range(0, len(images), batch_size):
            img_batch = images[b_idx:b_idx+batch_size]
            prompt_batch = prompts[b_idx:b_idx+batch_size]
            
            messages_list = [
                [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}]}]
                for img, txt in zip(img_batch, prompt_batch)
            ]

            try:
                text_inputs = [
                    self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
                    for msg in messages_list
                ]
                image_inputs, video_inputs = process_vision_info(messages_list)
                
                inputs = self.processor(
                    text=text_inputs, 
                    images=image_inputs, 
                    videos=video_inputs, 
                    padding=True, 
                    return_tensors="pt"
                ).to(self.model.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
                
                trimmed_ids = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                batch_captions = self.processor.batch_decode(trimmed_ids, skip_special_tokens=True)
                
                cleaned_captions = []
                for raw_text in batch_captions:
                    final_text = re.sub(r"<thinking>.*?</thinking>", "", raw_text, flags=re.DOTALL).strip()
                    answer_match = re.search(r"<answer>(.*?)</answer>", final_text, flags=re.DOTALL)
                    if answer_match:
                        cleaned_captions.append(answer_match.group(1).strip())
                    else:
                        cleaned_captions.append(final_text)
                
                all_captions.extend(cleaned_captions)
                
            except Exception as e:
                logger.error(f"Toplu vision islemi basarisiz: {e}", exc_info=True)
                all_captions.extend(["[Visual Description Failed]"] * len(img_batch))

        return all_captions