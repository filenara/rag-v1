import re
import json
import os

class STE100Guard:
    def __init__(self, json_path="config/ste100_rules_v9.json"):
        self.rules = self._load_json_rules(json_path)

    def _load_json_rules(self, path):
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
        Metni tarar, 37.000 kural içinden sadece bu metinde geçen kelimelerin 
        kurallarını filtreler ve LLM'e dinamik bir rehber (rapor) hazırlar.
        """
        feedback_report = []
        
        for rule in self.rules:
            if rule.get("is_approved") is False:
                keyword = rule.get("keyword", "")
                part_of_speech = rule.get("part_of_speech", "")
                
                # Regex burada sadece "Bu kelime metinde geçiyor mu?" diye hızlıca bakmak için var.
                pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                
                if pattern.search(text):
                    alts = ", ".join(rule.get("approved_alternatives", []))
                    desc = rule.get("rule_description", "")
                    
                    # LLM'in anlayacağı dilde, karar vermesini sağlayacak dinamik kural seti
                    report_item = f"- ŞÜPHELİ KELİME: '{keyword}'\n"
                    report_item += f"  > Kural: Bu kelimenin ({part_of_speech}) olarak kullanımı YASAKTIR.\n"
                    if alts:
                        report_item += f"  > Aksiyon: Eğer metinde {part_of_speech} olarak kullandıysan, anlamsal bütünlüğü bozmadan şu kelimelerle değiştir: {alts}. Eğer farklı bir formda (örn: isim) kullandıysan ve bu serbestse DEĞİŞTİRME.\n"
                    if desc:
                        report_item += f"  > Not: {desc}\n"
                        
                    feedback_report.append(report_item)
                    
        is_compliant = len(feedback_report) == 0
        return is_compliant, feedback_report