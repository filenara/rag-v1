from typing import List

class STE100SemanticSplitter:
    def __init__(self, normal_text_size_limit: float = 11.0):
        self.normal_text_size_limit = normal_text_size_limit

    def extract_semantic_chunks(self, page_dict: dict) -> List[str]:
        semantic_chunks = []
        current_heading = "Genel Baglam"
        current_content = ""

        blocks = page_dict.get("blocks", [])

        for b in blocks:
            if b.get("type") == 0:
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        
                        font_size = span.get("size", 0)
                        font_name = span.get("font", "").lower()

                        is_heading = font_size > self.normal_text_size_limit or "bold" in font_name

                        if is_heading:
                            if current_content.strip() and len(current_content) > 20:
                                chunk_text = f"[{current_heading}]\n{current_content.strip()}"
                                semantic_chunks.append(chunk_text)
                            
                            current_heading = text
                            current_content = ""
                        else:
                            current_content += text + " "
                
                current_content += "\n\n"

        if current_content.strip() and len(current_content) > 20:
            chunk_text = f"[{current_heading}]\n{current_content.strip()}"
            semantic_chunks.append(chunk_text)

        return semantic_chunks