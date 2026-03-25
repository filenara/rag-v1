import re
import logging
import spacy
from typing import List, Tuple

logger = logging.getLogger(__name__)


class STE100SemanticSplitter:
    def __init__(self, max_chunk_length: int = 1500):
        self.max_chunk_length = max_chunk_length
        self.current_hierarchy: List[Tuple[int, str]] = []
        self.carry_over_buffer = ""
        
        logger.info("spaCy (Sentencizer) yukleniyor...")
        try:
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")
            
            exceptions = ["Fig.", "Max.", "Min.", "Ref.", "Assy.", "Rev.", "No."]
            for exc in exceptions:
                self.nlp.tokenizer.add_special_case(exc, [{"ORTH": exc}])
        except Exception as e:
            logger.error("spaCy baslatilamadi: %s", e)
            self.nlp = None

    def _parse_heading(self, line: str) -> Tuple[int, str]:
        match = re.match(r'^(\d+(?:\.\d+)*)\.?\s+(.+)$', line.strip())
        if match:
            numbering = match.group(1)
            title = match.group(2)
            level = len(numbering.split('.'))
            return level, f"{numbering} {title}"
        return 0, ""

    def _update_hierarchy(self, level: int, title: str) -> None:
        self.current_hierarchy = [h for h in self.current_hierarchy if h[0] < level]
        self.current_hierarchy.append((level, title))

    def _get_context_prefix(self) -> str:
        if not self.current_hierarchy:
            return ""
        path = " > ".join([title for level, title in self.current_hierarchy])
        return f"[Context: {path}]\n"

    def _fallback_split(self, text: str, context_prefix: str) -> List[str]:
        if not self.nlp:
            return [f"{context_prefix}{text}"]
            
        doc = self.nlp(text)
        chunks = []
        current_chunk = ""
        
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            if not sentence_text:
                continue
                
            if len(current_chunk) + len(sentence_text) > self.max_chunk_length:
                if current_chunk:
                    chunks.append(f"{context_prefix}{current_chunk.strip()}")
                current_chunk = sentence_text
            else:
                current_chunk += " " + sentence_text
                
        if current_chunk:
            chunks.append(f"{context_prefix}{current_chunk.strip()}")
            
        return chunks

    def extract_semantic_chunks(self, raw_text: str, is_last_page: bool = False) -> List[str]:
        combined_text = f"{self.carry_over_buffer}\n{raw_text}".strip()
        self.carry_over_buffer = ""

        if not combined_text:
            return []

        lines = combined_text.split('\n')
        
        blocks = []
        current_block_text = ""
        
        for line in lines:
            level, title = self._parse_heading(line)
            
            if level > 0:
                if current_block_text.strip():
                    blocks.append((self._get_context_prefix(), current_block_text.strip()))
                    current_block_text = ""
                    
                self._update_hierarchy(level, title)
            else:
                current_block_text += line + " "

        if current_block_text.strip():
            if is_last_page:
                blocks.append((self._get_context_prefix(), current_block_text.strip()))
            else:
                if not re.search(r'(?:[.!?]|\n{2,})\s*$', current_block_text):
                    self.carry_over_buffer = current_block_text
                else:
                    blocks.append((self._get_context_prefix(), current_block_text.strip()))

        final_chunks = []
        for prefix, block_text in blocks:
            if len(block_text) > self.max_chunk_length:
                final_chunks.extend(self._fallback_split(block_text, prefix))
            else:
                final_chunks.append(f"{prefix}{block_text}")

        return final_chunks

    def reset_buffer(self) -> None:
        self.carry_over_buffer = ""
        self.current_hierarchy = []