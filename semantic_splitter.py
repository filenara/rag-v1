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
        logger.info("Dokuman hiyerarsik ve eleman bazli izole ediliyor...")
        chunks = []
        
        try:
            doc_chunks = list(self.chunker.chunk(document))
            
            for chunk in doc_chunks:
                if not hasattr(chunk.meta, "doc_items") or not chunk.meta.doc_items:
                    continue
                    
                headings = chunk.meta.headings if hasattr(chunk.meta, "headings") and chunk.meta.headings else []
                parent_context = " > ".join(headings)
                
                current_text_buffer = []
                last_page_no = 0
                
                def flush_text_buffer():
                    """Tampon bellekte biriken standart metinleri tek bir parca olarak disari aktarir."""
                    if current_text_buffer:
                        metadata = {
                            "source": source_name,
                            "parent_context": parent_context,
                            "page": last_page_no
                        }
                        chunks.append({
                            "text": "\n".join(current_text_buffer),
                            "metadata": metadata
                        })
                        current_text_buffer.clear()
                
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
                            
                flush_text_buffer()
                
            logger.info("Toplam %d adet izole edilmis parca olusturuldu.", len(chunks))
            return chunks
        
        except Exception as e:
            logger.error("Parcalama islemi sirasinda hata: %s", e)
            return []