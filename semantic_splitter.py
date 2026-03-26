import logging
from typing import List, Dict, Any
from docling.chunking import HierarchicalChunker

logger = logging.getLogger(__name__)


class STE100SemanticSplitter:
    def __init__(self, max_chunk_length: int = 1500, **kwargs):
        # max_chunk_length geriye donuk uyumluluk (orchestrator kirmamak) icin tutulmustur.
        self.max_chunk_length = max_chunk_length
        logger.info("Docling HierarchicalChunker baslatiliyor...")
        self.chunker = HierarchicalChunker()

    def extract_semantic_chunks(self, document: Any, source_name: str = "Unknown") -> List[Dict[str, Any]]:
        logger.info("Dokuman hiyerarsik olarak parcalaniyor...")
        chunks = []
        
        try:
            doc_chunks = list(self.chunker.chunk(document))
            
            for chunk in doc_chunks:
                text_content = chunk.text.strip()
                if not text_content:
                    continue
                
                # Docling uzerinden hiyerarsik basliklari cekme
                headings = chunk.meta.headings if hasattr(chunk.meta, "headings") and chunk.meta.headings else []
                parent_context = " > ".join(headings)
                
                # Sayfa numarasini guvenli sekilde alma
                page_no = 0
                if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                    prov = chunk.meta.doc_items[0].prov
                    if prov:
                        page_no = prov[0].page_no
                
                metadata = {
                    "source": source_name,
                    "parent_context": parent_context,
                    "page": page_no
                }
                
                chunks.append({
                    "text": text_content,
                    "metadata": metadata
                })
                
            logger.info("Toplam %d adet hiyerarsik parca olusturuldu.", len(chunks))
            return chunks
        
        except Exception as e:
            logger.error("Parcalama islemi sirasinda hata: %s", e)
            return []