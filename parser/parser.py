import pdfplumber
import json
import re
import logging
from typing import Dict, Any, Optional, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class STE100Parser:
    def __init__(self, pdf_path: str, start_page: int, end_page: int):
        self.pdf_path = pdf_path
        self.start_page = start_page
        self.end_page = end_page
        self.parsed_dictionary: Dict[str, Any] = {}
        self.current_word_data: Optional[Dict[str, str]] = None
        
        # Kati Sutun Matrisi (Strict Column Boundaries)
        self.col1_max_x = 140
        self.col2_max_x = 265
        
        self.pos_pattern = r'\((v|n|adj|adv|prep|conj|pron|art|pn)\)'

    def is_unwanted_line(self, text: str) -> bool:
        """Satirin baslik, alt bilgi veya tablo kalintisi olup olmadigini kontrol eder."""
        if not text:
            return True
            
        text_upper = text.upper().strip()
        unwanted_exact = [
            "SIMPLIFIED TECHNICAL ENGLISH",
            "ASD-STE100",
            "WORD APPROVED MEANING/",
            "(PART OF SPEECH)",
            "ALTERNATIVES",
            "APPROVED EXAMPLE",
            "NOT APPROVED",
            "BLANK PAGE"
        ]
        
        for unwanted in unwanted_exact:
            if unwanted in text_upper:
                return True
                
        if re.search(r'\b\d{4}-\d{2}-\d{2}\b', text_upper):
            return True
        if re.search(r'ISSUE \d+', text_upper):
            return True
        if re.search(r'PAGE \d+', text_upper) or re.search(r'PAGE [A-Z0-9-]+', text_upper):
            return True
            
        return False

    def extract_page_lines(self, page: pdfplumber.page.Page) -> List[Tuple[str, str]]:
        """Sayfadaki kelimeleri X eksenindeki kesin koordinatlarina gore 2 havuza ayirir."""
        words = page.extract_words(keep_blank_chars=False)
        lines_dict: Dict[float, List[Dict[str, Any]]] = {}
        
        page_height = float(page.height)
        top_margin = page_height * 0.08
        bottom_margin = page_height * 0.92
        
        for word in words:
            if word["top"] < top_margin or word["bottom"] > bottom_margin:
                continue
                
            top = round(word["top"], 1)
            matched_top = None
            
            for existing_top in lines_dict.keys():
                if abs(existing_top - top) < 3.0:
                    matched_top = existing_top
                    break
                    
            if matched_top is None:
                matched_top = top
                lines_dict[matched_top] = []
                
            lines_dict[matched_top].append(word)
            
        sorted_tops = sorted(lines_dict.keys())
        page_data = []
        
        for top in sorted_tops:
            line_words = lines_dict[top]
            line_words.sort(key=lambda x: x["x0"])
            
            if not line_words:
                continue
            
            full_line_text = " ".join([w["text"] for w in line_words])
            if self.is_unwanted_line(full_line_text):
                continue
                
            # X > 310 olan kelimeleri dogrudan yoksay (3. ve 4. Kolon reddi)
            c1_parts = [w["text"] for w in line_words if w["x0"] < self.col1_max_x]
            c2_parts = [w["text"] for w in line_words if self.col1_max_x <= w["x0"] < self.col2_max_x]
            
            c1_text = " ".join(c1_parts).strip()
            c2_text = " ".join(c2_parts).strip()
            
            if c1_text or c2_text:
                page_data.append((c1_text, c2_text))
                
        return page_data

    def has_pos_tag(self, text: str) -> bool:
        """Metin icinde gecerli bir tur etiketi (POS) olup olmadigini kontrol eder."""
        return bool(re.search(self.pos_pattern, text))

    def process_word_data(self) -> None:
        """Biriktirilen temiz kelime verisini sozluge kaydeder."""
        if not self.current_word_data:
            return
            
        c1_raw = self.current_word_data["c1"].strip()
        c2_raw = self.current_word_data["c2"].strip()
        
        if not c1_raw:
            self.current_word_data = None
            return

        pos_match = re.search(r'^([A-Za-z\s-]+)\s*' + self.pos_pattern, c1_raw)
        if not pos_match:
            self.current_word_data = None
            return
            
        primary_word = pos_match.group(1).strip()
        part_of_speech = pos_match.group(2).strip()
        is_approved = primary_word.isupper()
        
        forms_set = {primary_word}
        forms_list = [primary_word]
        
        rest_c1 = c1_raw[pos_match.end():].strip()
        if rest_c1:
            extra_words = re.findall(r'\b[A-Z]+\b', rest_c1)
            for ew in extra_words:
                if len(ew) > 1 and ew not in forms_set:
                    forms_set.add(ew)
                    forms_list.append(ew)

        note_text = None
        main_c2 = c2_raw
        note_match = re.search(r'\bNOTE:(.*)', c2_raw, flags=re.IGNORECASE | re.DOTALL)
        
        if note_match:
            note_text = note_match.group(1).strip()
            main_c2 = c2_raw[:note_match.start()].strip()
            
        meaning = None
        approved_alternatives = []
        
        if is_approved:
            meaning = main_c2 if main_c2 else None
        else:
            alt_pattern = r'([A-Z][A-Z\s-]+)\s*' + self.pos_pattern
            alt_matches = re.finditer(alt_pattern, main_c2)
            found_alts = [m.group(0).strip() for m in alt_matches]
            
            if found_alts:
                approved_alternatives = found_alts
            elif main_c2:
                if note_text:
                    note_text = main_c2 + " " + note_text
                else:
                    note_text = main_c2

        root_keyword = primary_word.lower()
        
        if root_keyword not in self.parsed_dictionary:
            self.parsed_dictionary[root_keyword] = {
                "keyword": root_keyword,
                "forms": [],
                "notes": note_text
            }
        else:
            existing_note = self.parsed_dictionary[root_keyword].get("notes")
            if note_text:
                if existing_note:
                    self.parsed_dictionary[root_keyword]["notes"] = existing_note + " " + note_text
                else:
                    self.parsed_dictionary[root_keyword]["notes"] = note_text

        for form in forms_list:
            form_entry = {
                "word_form": form,
                "part_of_speech": part_of_speech,
                "is_approved": is_approved,
                "approved_alternatives": approved_alternatives.copy(),
                "meaning": meaning
            }
            self.parsed_dictionary[root_keyword]["forms"].append(form_entry)
            
        self.current_word_data = None

    def extract_data(self) -> None:
        logger.info(f"PDF okuma baslatiliyor: {self.pdf_path}")
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                total_pages = len(pdf.pages)
                safe_end = min(self.end_page, total_pages)
                
                for page_num in range(self.start_page - 1, safe_end):
                    page = pdf.pages[page_num]
                    page_lines = self.extract_page_lines(page)
                    
                    for c1, c2 in page_lines:
                        if self.has_pos_tag(c1):
                            self.process_word_data()
                            self.current_word_data = {"c1": c1, "c2": c2}
                        else:
                            if self.current_word_data:
                                if c1:
                                    self.current_word_data["c1"] += " " + c1
                                if c2:
                                    self.current_word_data["c2"] += " " + c2
                                    
                    if (page_num + 1) % 10 == 0:
                        logger.info(f"Islenen sayfa: {page_num + 1}/{safe_end}")
                        
                self.process_word_data()
                
        except Exception as e:
            logger.error(f"PDF okunurken hata olustu: {e}")

    def export_to_json(self, output_path: str) -> None:
        if not self.parsed_dictionary:
            logger.warning("Disa aktarilacak veri bulunamadi.")
            return

        final_list = list(self.parsed_dictionary.values())
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_list, f, ensure_ascii=False, indent=4)
            logger.info(f"Veriler basariyla {output_path} dosyasina kaydedildi. Toplam kelime grubu: {len(final_list)}")
        except Exception as e:
            logger.error(f"JSON kaydetme hatasi: {e}")


if __name__ == "__main__":
    PDF_FILE_PATH = "../data/ASD-STE100-ISSUE-7.pdf"
    OUTPUT_JSON_PATH = "../config/ste100_rules.json"
    
    START_PAGE = 103
    END_PAGE = 390
    
    parser = STE100Parser(
        pdf_path=PDF_FILE_PATH, 
        start_page=START_PAGE, 
        end_page=END_PAGE
    )
    
    parser.extract_data()
    parser.export_to_json(OUTPUT_JSON_PATH)