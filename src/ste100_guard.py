import re
import json
import os

class STE100Guard:
    def __init__(self, json_path="config/ste100_rules.json"):
        """
        STE100 denetleyicisini baslatir. 
        Varsayilan olarak v9 kural seti kullanilir.
        """
        self.rules = self._load_json_rules(json_path)

    def _load_json_rules(self, path):
        """JSON formatindaki kurallari guvenli sekilde yukler."""
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"JSON Okuma Hatasi: {e}")
        else:
            print(f"Uyari: {path} bulunamadi.")
        return []

    def analyze_and_report(self, text):
        """
        Metni tarar ve metinde gecen yasakli kelimeler icin 
        modelin anlayabilecegi dinamik bir rapor hazirlar.
        """
        feedback_report = []
        
        for rule in self.rules:
            if rule.get("is_approved") is False:
                keyword = rule.get("keyword", "")
                part_of_speech = rule.get("part_of_speech", "")
                
                # Kelimenin tam eslesmesi icin regex kullanimi
                pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                
                if pattern.search(text):
                    alts = ", ".join(rule.get("approved_alternatives", []))
                    desc = rule.get("rule_description", "")
                    
                    report_item = f"- SUPHELI KELIME: '{keyword}'\n"
                    report_item += f"  > Kural: Bu kelimenin ({part_of_speech}) olarak kullanimi YASAKTIR.\n"
                    if alts:
                        report_item += f"  > Aksiyon: Eger metinde {part_of_speech} olarak kullandiysan, sunlarla degistir: {alts}.\n"
                    if desc:
                        report_item += f"  > Not: {desc}\n"
                        
                    feedback_report.append(report_item)
                    
        is_compliant = len(feedback_report) == 0
        return is_compliant, feedback_report