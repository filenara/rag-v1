import re
from src.utils import load_ste100_rules

class STE100Guard:
    def __init__(self):
        self.rules = load_ste100_rules()

    def check_compliance(self, text):
        """Metni tarar ve STE100 ihlallerini raporlar."""
        warnings = []
        forbidden_map = self.rules.get("forbidden_words", {})
        
        # Basit kelime taraması
        for forbidden, suggested in forbidden_map.items():
            # Regex: Kelimenin tam hali (büyük küçük harf duyarsız)
            pattern = re.compile(r'\b' + re.escape(forbidden) + r'\b', re.IGNORECASE)
            
            if pattern.search(text):
                warnings.append(f"⚠️ STE100 İhlali: '{forbidden}' yerine '{suggested}' kullanılmalı.")
        
        return warnings

    def apply_corrections(self, text):
        """Yasaklı kelimeleri otomatik düzeltir (Opsiyonel)."""
        corrected_text = text
        forbidden_map = self.rules.get("forbidden_words", {})
        
        for forbidden, suggested in forbidden_map.items():
            pattern = re.compile(r'\b' + re.escape(forbidden) + r'\b', re.IGNORECASE)
            corrected_text = pattern.sub(suggested, corrected_text)
            
        return corrected_text