import io
import os
import hashlib
import logging
from typing import List
import fitz
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class VisualAsset(BaseModel):
    image: Image.Image
    path: str

    class Config:
        arbitrary_types_allowed = True


class DocumentParser:
    def __init__(self, assets_dir: str, min_width: int = 80, min_height: int = 80):
        self.assets_dir = assets_dir
        self.min_width = min_width
        self.min_height = min_height
        self.image_dpi_scale = 2.0
        self.vector_spam_limit = 600
        self.normal_text_size_limit = 11.0
        os.makedirs(self.assets_dir, exist_ok=True)

    def open_document(self, pdf_path: str) -> fitz.Document:
        return fitz.open(pdf_path)

    def extract_text(self, page: fitz.Page) -> str:
        blocks = page.get_text("dict")["blocks"]
        extracted_text = ""

        for block in blocks:
            if block["type"] == 0:
                block_text = ""
                is_heading = False
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        
                        font_size = span["size"]
                        font_name = span["font"].lower()

                        if font_size > self.normal_text_size_limit or "bold" in font_name:
                            is_heading = True
                        
                        block_text += text + " "
                
                block_text = block_text.strip()
                if block_text:
                    if is_heading:
                        extracted_text += f"\n{block_text}\n"
                    else:
                        extracted_text += f"{block_text}\n"
                        
        return extracted_text.strip()

    def extract_text_dict(self, page: fitz.Page) -> dict:
        return page.get_text("dict")

    def extract_visuals(self, page: fitz.Page, filename: str, page_num: int) -> List[VisualAsset]:
        all_visual_rects = []
        
        for img in page.get_images(full=True):
            rect = page.get_image_bbox(img)
            if rect.width > 50 and rect.height > 50:
                all_visual_rects.append(list(rect))
                
        paths = page.get_drawings()
        if len(paths) <= self.vector_spam_limit:
            for path in paths:
                rect = path["rect"]
                if rect.width > 5 or rect.height > 5:
                    all_visual_rects.append(list(rect))

        text_blocks = page.get_text("blocks")
        for block in text_blocks:
            if len(block[4]) < 200:
                all_visual_rects.append([block[0], block[1], block[2], block[3]])

        merged_rects = self._merge_boxes(all_visual_rects)
        assets = []
        
        for rect in merged_rects:
            w, h = rect[2] - rect[0], rect[3] - rect[1]
            if w < self.min_width or h < self.min_height:
                continue
            
            try:
                clip_rect = fitz.Rect(rect[0], rect[1], rect[2], rect[3])
                mat = fitz.Matrix(self.image_dpi_scale, self.image_dpi_scale)
                pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                img_data = pix.tobytes("png")
                pil_img = Image.open(io.BytesIO(img_data))
                
                img_hash = hashlib.md5(pil_img.tobytes()).hexdigest()
                asset_name = f"vis_{img_hash}.png"
                save_path = os.path.join(self.assets_dir, asset_name)
                
                if not os.path.exists(save_path):
                    pix.save(save_path)
                
                assets.append(VisualAsset(image=pil_img, path=save_path))
            except OSError as e:
                logger.warning(
                    "Gorsel diske kaydedilemedi (Sayfa: %s): %s", 
                    page_num, e, 
                    exc_info=True
                )
                continue
            except Exception as e:
                logger.warning(
                    "Gorsel cikarimi sirasinda beklenmeyen hata (Sayfa: %s): %s", 
                    page_num, e, 
                    exc_info=True
                )
                continue
                
        return assets

    def _merge_boxes(self, boxes: List[List[float]]) -> List[List[float]]:
        if not boxes:
            return []
        merged = []
        while boxes:
            current = boxes.pop(0)
            has_overlap = True
            while has_overlap:
                has_overlap = False
                rest = []
                for other in boxes:
                    if self._boxes_intersect_or_close(current, other):
                        current = [
                            min(current[0], other[0]),
                            min(current[1], other[1]),
                            max(current[2], other[2]),
                            max(current[3], other[3])
                        ]
                        has_overlap = True
                    else:
                        rest.append(other)
                boxes = rest
            merged.append(current)
        return merged

    def _boxes_intersect_or_close(self, box1: List[float], box2: List[float], margin: int = 150) -> bool:
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        return not (x1_max + margin < x2_min or x1_min - margin > x2_max or
                    y1_max + margin < y2_min or y1_min - margin > y2_max)