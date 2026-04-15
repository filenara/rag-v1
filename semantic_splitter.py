import logging
from typing import List, Dict, Any
from docling.chunking import HierarchicalChunker

logger = logging.getLogger(__name__)


class STE100SemanticSplitter:
    def __init__(self, max_chunk_length: int = 1500, **kwargs):
        self.max_chunk_length = max_chunk_length
        logger.info("Docling HierarchicalChunker baslatiliyor...")
        self.chunker = HierarchicalChunker()

    def extract_semantic_chunks(self, document: Any, source_name: str = "Unknown") -> List[Dict[str, Any]]:
        logger.info("Dokuman hiyerarsik ve eleman bazli izole ediliyor (Tekillestirme aktif)...")
        chunks = []
        
        try:
            doc_chunks = list(self.chunker.chunk(document))
            
            current_text_buffer = []
            overlap_buffer = ""
            last_page_no = 0
            parent_context = ""
            processed_visuals = set()
            
            def flush_text_buffer():
                nonlocal overlap_buffer
                
                if current_text_buffer:
                    if len(current_text_buffer) == 1 and current_text_buffer[0] == overlap_buffer:
                        return
                        
                    metadata = {
                        "source": source_name,
                        "parent_context": parent_context,
                        "page": last_page_no
                    }
                    
                    chunks.append({
                        "text": "\n".join(current_text_buffer),
                        "metadata": metadata
                    })
                    
                    overlap_buffer = current_text_buffer[-1]
                    current_text_buffer.clear()
                    current_text_buffer.append(overlap_buffer)

            for chunk in doc_chunks:
                if not hasattr(chunk.meta, "doc_items") or not chunk.meta.doc_items:
                    continue
                    
                headings = chunk.meta.headings if hasattr(chunk.meta, "headings") and chunk.meta.headings else []
                new_parent_context = " > ".join(headings)
                
                if new_parent_context != parent_context:
                    flush_text_buffer()
                    parent_context = new_parent_context
                
                for item in chunk.meta.doc_items:
                    item_type = type(item).__name__
                    
                    page_no = 0
                    if hasattr(item, "prov") and item.prov:
                        page_no = item.prov[0].page_no
                        last_page_no = page_no
                        
                    metadata = {
                        "source": source_name,
                        "parent_context": parent_context,
                        "page": page_no
                    }
                    
                    if item_type == "PictureItem":
                        visual_identifier = None
                        if hasattr(item, "image") and hasattr(item.image, "uri") and item.image.uri:
                            visual_identifier = item.image.uri
                        else:
                            temp_text = item.text.strip() if hasattr(item, "text") and item.text else ""
                            if temp_text:
                                visual_identifier = hash(temp_text)
                                
                        if visual_identifier:
                            if visual_identifier in processed_visuals:
                                continue
                            processed_visuals.add(visual_identifier)

                        flush_text_buffer()
                        text_content = item.text.strip() if hasattr(item, "text") and item.text else ""
                        
                        if not text_content:
                            text_content = "[Gorsel Icerik]"
                            
                        if hasattr(item, "image") and hasattr(item.image, "uri") and item.image.uri:
                            metadata["image_path"] = item.image.uri
                        else:
                            metadata["has_visual"] = "True"
                            
                        chunks.append({
                            "text": text_content,
                            "metadata": metadata
                        })
                        
                    elif item_type == "TableItem":
                        flush_text_buffer()
                        if hasattr(item, "export_to_markdown"):
                            text_content = item.export_to_markdown()
                        else:
                            text_content = item.text.strip() if hasattr(item, "text") and item.text else ""
                            
                        if text_content:
                            chunks.append({
                                "text": text_content,
                                "metadata": metadata
                            })
                            
                    elif item_type in ["TextItem", "SectionHeaderItem", "ListItem"]:
                        text_val = item.text.strip() if hasattr(item, "text") and item.text else ""
                        if text_val:
                            current_text_buffer.append(text_val)
                            
            current_text_buffer.clear()
            
            logger.info("Toplam %d adet izole edilmis parca olusturuldu.", len(chunks))
            return chunks
        
        except Exception as e:
            logger.error("Parcalama islemi sirasinda hata: %s", e)
            return []