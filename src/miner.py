import fitz 
import io
from PIL import Image

class CompositeVisualMiner:
    def __init__(self, min_width=100, min_height=100, vector_spam_limit=600):
        self.min_width = min_width
        self.min_height = min_height
        self.vector_spam_limit = vector_spam_limit 

    def _boxes_intersect_or_close(self, box1, box2, margin=10): 
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        return not (x1_max + margin < x2_min or x1_min - margin > x2_max or
                    y1_max + margin < y2_min or y1_min - margin > y2_max)

    def _merge_boxes(self, boxes):
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

    def extract_visual_crops(self, page):
        all_visual_rects = []
        
        # Resimleri al
        images = page.get_images(full=True)
        for img in images:
            rect = page.get_image_bbox(img)
            if (rect.width > 50 and rect.height > 50):
                all_visual_rects.append(list(rect))
                
        # Çizimleri (Tabloları) al
        paths = page.get_drawings()
        if len(paths) < self.vector_spam_limit:
            for path in paths:
                rect = path["rect"]
                if rect.width > 5 or rect.height > 5:
                    all_visual_rects.append(list(rect))

        if not all_visual_rects: return []

        merged_rects = self._merge_boxes(all_visual_rects)
        final_crops = []
        
        for rect in merged_rects:
            w, h = rect[2]-rect[0], rect[3]-rect[1]
            if w < self.min_width or h < self.min_height: continue
            try:
                clip_rect = fitz.Rect(rect[0], rect[1], rect[2], rect[3])
                zoom_factor = 2.5 
                mat = fitz.Matrix(zoom_factor, zoom_factor)
                pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                img_data = pix.tobytes("png")
                final_crops.append(Image.open(io.BytesIO(img_data)))
            except: 
                pass
        return final_crops