import fitz  # PyMuPDF
import io
from PIL import Image
from config.settings import cfg
from qwen_vl_utils import process_vision_info
import torch

class PDFMiner:
    def __init__(self):
        self.min_width = cfg.MIN_IMAGE_DIM
        self.min_height = cfg.MIN_IMAGE_DIM
        self.vector_spam_limit = cfg.VECTOR_SPAM_LIMIT

    def _boxes_intersect_or_close(self, box1, box2, margin=150):
        # Kutuların çakışıp çakışmadığını kontrol eder
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        return not (x1_max + margin < x2_min or x1_min - margin > x2_max or
                    y1_max + margin < y2_min or y1_min - margin > y2_max)

    def _merge_boxes(self, boxes):
        # Yakın kutuları birleştirir (Görsel bütünlüğü için)
        if not boxes: return []
        merged = []
        while boxes:
            current = boxes.pop(0)
            has_overlap = True
            while has_overlap:
                has_overlap = False
                rest = []
                for other in boxes:
                    if self._boxes_intersect_or_close(current, other):
                        current = (
                            min(current[0], other[0]),
                            min(current[1], other[1]),
                            max(current[2], other[2]),
                            max(current[3], other[3])
                        )
                        has_overlap = True
                    else:
                        rest.append(other)
                boxes = rest
            merged.append(current)
        return merged

    def extract_images_from_page(self, page):
        """Bir sayfadaki görselleri ve tabloları tespit edip kırpar."""
        all_visual_rects = []
        
        # 1. Resimleri bul
        for img in page.get_images(full=True):
            rect = page.get_image_bbox(img)
            if rect.width > 50 and rect.height > 50:
                all_visual_rects.append(list(rect))

        # 2. Çizimleri (Vektör grafikler/Tablolar) bul
        paths = page.get_drawings()
        if len(paths) <= self.vector_spam_limit:
            for path in paths:
                rect = path["rect"]
                if rect.width > 5 or rect.height > 5:
                    all_visual_rects.append(list(rect))
        
        if not all_visual_rects: return []

        # Kutuları birleştir
        merged_rects = self._merge_boxes(all_visual_rects)
        final_crops = []

        for rect in merged_rects:
            w, h = rect[2]-rect[0], rect[3]-rect[1]
            if w < self.min_width or h < self.min_height: continue
            
            try:
                clip_rect = fitz.Rect(rect[0], rect[1], rect[2], rect[3])
                pix = page.get_pixmap(dpi=cfg.IMAGE_DPI, clip=clip_rect)
                img_data = pix.tobytes("png")
                final_crops.append(Image.open(io.BytesIO(img_data)))
            except Exception as e:
                print(f"Görsel kırpma hatası: {e}") # İlerde buraya logger gelecek
                pass
        
        return final_crops

def generate_caption(pil_image, model, processor):
    """Qwen-VL kullanarak görsele açıklama yazar."""
    prompt = "Describe this image in detail for a blind person. Read all text verbatim."
    
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": prompt}
        ]}
    ]
    
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text_input], 
        images=image_inputs, 
        videos=video_inputs, 
        padding=True, 
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]