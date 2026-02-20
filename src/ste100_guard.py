import re
import json
import os

class STE100Guard:
    def __init__(self, json_path="config/ste100_rules_v9.json"):
        self.rules = self._load_json_rules(json_path)

    def _load_json_rules(self, path):
        """Senin hazırladığın v9 sözlüğünü yükler."""
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"JSON Okuma Hatası: {e}")
        else:
            print(f"Uyarı: {path} bulunamadı.")
        return []

    def analyze_and_report(self, text):
        """
        Metni tarar ve LLM'in kendini düzeltmesi için detaylı, 
        bağlam (in-context) içeren bir geri bildirim (feedback) raporu üretir.
        """
        feedback_report = []
        
        for rule in self.rules:
            # Sadece onaylı olmayan (is_approved: false) kelimeleri arıyoruz
            if rule.get("is_approved") is False:
                keyword = rule.get("keyword", "")
                part_of_speech = rule.get("part_of_speech", "")
                
                # Regex sadece 'tespit' için kullanılıyor, 'değiştirme' için değil.
                pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                
                if pattern.search(text):
                    # LLM'i eğitecek verileri JSON'dan çek
                    alts = ", ".join(rule.get("approved_alternatives", []))
                    desc = rule.get("rule_description", "")
                    
                    # Prompt'a gidecek rapor maddesini oluştur
                    report_item = f"- YASAKLI KELİME: '{keyword}' ({part_of_speech})\n"
                    if alts:
                        report_item += f"  > Bunun Yerine Kullan: {alts}\n"
                    if desc:
                        report_item += f"  > Kural Notu: {desc}\n"
                    
                    # JSON'daki örneği (in-context learning için) ekle
                    examples = rule.get("examples", [])
                    if examples and examples[0].get("ste"):
                        report_item += f"  > Örnek Doğru Kullanım: {examples[0]['ste']}\n"
                        
                    feedback_report.append(report_item)
                    
        is_compliant = len(feedback_report) == 0
        return is_compliant, feedback_report