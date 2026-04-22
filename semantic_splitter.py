import logging
from typing import List, Dict, Any
from docling.chunking import HierarchicalChunker

logger = logging.getLogger(__name__)


class STE100SemanticSplitter:
    def __init__(self, max_chunk_length: int = 1500, chunk_overlap: int = 200, **kwargs):
        self.max_chunk_length = max_chunk_length
        self.chunk_overlap = chunk_overlap
        logger.info("Docling HierarchicalChunker baslatiliyor...")
        self.chunker = HierarchicalChunker()

    def extract_semantic_chunks(self, document: Any, source_name: str = "Unknown") -> List[Dict[str, Any]]:
        logger.info("Dokuman hiyerarsik ve eleman bazli izole ediliyor (Sizinti onleme aktif)...")
        chunks = []
        
        try:
            doc_chunks = list(self.chunker.chunk(document))
            
            current_text_buffer = []
            current_length = 0
            overlap_buffer = ""
            last_page_no = 0
            parent_context = ""
            
            def flush_text_buffer(force_clear_overlap: bool = False):
                nonlocal overlap_buffer, current_length, current_text_buffer
                
                if not current_text_buffer:
                    return

                combined_text = "\n".join(current_text_buffer).strip()
                
                if combined_text == overlap_buffer.strip() and not force_clear_overlap:
                    return
                    
                metadata = {
                    "source": source_name,
                    "parent_context": parent_context,
                    "page": last_page_no
                }
                
                chunks.append({
                    "text": combined_text,
                    "metadata": metadata
                })
                
                if force_clear_overlap:
                    overlap_buffer = ""
                else:
                    if len(combined_text) > self.chunk_overlap:
                        overlap_buffer = combined_text[-self.chunk_overlap:]
                    else:
                        overlap_buffer = combined_text
                    
                current_text_buffer.clear()
                current_length = 0
                
                if overlap_buffer and not force_clear_overlap:
                    current_text_buffer.append(f"... {overlap_buffer}")
                    current_length = len(current_text_buffer[0])

            for chunk in doc_chunks:
                if not hasattr(chunk.meta, "doc_items") or not chunk.meta.doc_items:
                    continue
                    
                headings = chunk.meta.headings if hasattr(chunk.meta, "headings") and chunk.meta.headings else []
                new_parent_context = " > ".join(headings)
                
                if new_parent_context != parent_context:
                    flush_text_buffer(force_clear_overlap=True)
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
                    
                    if item_type == "TableItem":
                        flush_text_buffer(force_clear_overlap=True)
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
                            if current_length + len(text_val) > self.max_chunk_length:
                                flush_text_buffer()
                                
                            current_text_buffer.append(text_val)
                            current_length += len(text_val)
                            
            flush_text_buffer(force_clear_overlap=True)
            
            logger.info("Toplam %d adet izole edilmis parca olusturuldu.", len(chunks))
            return chunks
        
        except Exception as e:
            logger.error("Parcalama islemi sirasinda hata: %s", e)
            return []