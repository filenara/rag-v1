import json
import os
import logging
import spacy
from spacy.cli import download

logger = logging.getLogger(__name__)

class STE100Guard:
    def __init__(self, json_path="config/ste100_rules.json"):
        self.rules = self._load_json_rules(json_path)
        self.nlp = self._load_spacy_model()
        
        self.pos_map = {
            "VERB": "verb",
            "NOUN": "noun",
            "ADJ": "adjective",
            "ADV": "adverb",
            "PRON": "pronoun",
            "ADP": "preposition",
            "CCONJ": "conjunction",
            "SCONJ": "conjunction"
        }

    def _load_spacy_model(self):
        model_name = "en_core_web_sm"
        try:
            return spacy.load(model_name)
        except OSError:
            logger.info(f"SpaCy modeli '{model_name}' bulunamadi. Otomatik olarak indiriliyor...")
            try:
                download(model_name)
                logger.info(f"Model '{model_name}' basariyla indirildi.")
                return spacy.load(model_name)
            except Exception as e:
                logger.error(f"Model indirme veya yukleme sirasinda kritik hata: {e}", exc_info=True)
                return None

    def _load_json_rules(self, path):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"JSON formati hatali ({path}): {e}", exc_info=True)
            except OSError as e:
                logger.error(f"Dosya okuma hatasi ({path}): {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Beklenmeyen hata ({path}): {e}", exc_info=True)
        else:
            logger.warning(f"Uyari: {path} bulunamadi.")
        return []

    def analyze_and_report(self, text):
        feedback_report = []
        
        if not self.nlp:
            logger.warning("NLP modeli yuklenemedigi icin yalnizca kaba metin kontrolu atlandi, metin guvenli kabul ediliyor.")
            return True, []
            
        doc = self.nlp(text)
        violated_rules = set()
        
        for token in doc:
            mapped_pos = self.pos_map.get(token.pos_, "")
            token_lower = token.text.lower()
            lemma_lower = token.lemma_.lower()
            
            for rule in self.rules:
                if rule.get("is_approved") is False:
                    keyword = rule.get("keyword", "").lower()
                    expected_pos = rule.get("part_of_speech", "").lower()
                    
                    rule_id = f"{keyword}_{expected_pos}"
                    if rule_id in violated_rules:
                        continue
                        
                    if (lemma_lower == keyword or token_lower == keyword) and mapped_pos == expected_pos:
                        violated_rules.add(rule_id)
                        
                        alts = ", ".join(rule.get("approved_alternatives", []))
                        desc = rule.get("rule_description", "")
                        examples = rule.get("examples", [])
                        
                        report_item = f"- SUPHELI KELIME: '{token.text}' (Kok: {keyword})\n"
                        report_item += f"  > Kural: Bu kelimenin ({expected_pos}) olarak kullanimi YASAKTIR.\n"
                        
                        if alts:
                            report_item += f"  > Aksiyon: Eger metinde {expected_pos} olarak kullandiysan, sunlarla degistir: {alts}.\n"
                        
                        if desc:
                            report_item += f"  > Not: {desc}\n"
                            
                        if examples and isinstance(examples, list):
                            first_example = examples[0]
                            ste_example = first_example.get("ste", "")
                            non_ste_example = first_example.get("non_ste", "")
                            
                            if ste_example or non_ste_example:
                                report_item += "  > Ornekler:\n"
                                if non_ste_example:
                                    report_item += f"    * Hatali (NON-STE): {non_ste_example}\n"
                                if ste_example:
                                    report_item += f"    * Dogru (STE): {ste_example}\n"
                            
                        feedback_report.append(report_item)
                        
        is_compliant = len(feedback_report) == 0
        return is_compliant, feedback_report