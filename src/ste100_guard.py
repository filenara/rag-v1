import os
import json
import logging
import spacy
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

_SPACY_MODEL = None


class STE100Guard:
    def __init__(
        self, 
        dictionary_path: str = "config/ste100_rules.json", 
        core_rules_path: str = "config/ste100_core_rules.json"
    ):
        global _SPACY_MODEL
        
        self.dictionary_path = dictionary_path
        self.core_rules_path = core_rules_path
        
        self.dictionary_rules = self._load_json(self.dictionary_path, "rules")
        self.core_rules = self._load_json_list(self.core_rules_path, "core_rules")
        
        if _SPACY_MODEL is None:
            logger.info("STE100Guard icin spaCy (en_core_web_sm) bellege aliniyor...")
            try:
                _SPACY_MODEL = spacy.load("en_core_web_sm")
            except OSError as e:
                error_msg = (
                    "Kritik Hata: 'en_core_web_sm' modeli bulunamadi. "
                    "STE100 denetimi yapilamaz. Lutfen terminalde su komutu calistirin: "
                    "python -m spacy download en_core_web_sm"
                )
                logger.critical(error_msg)
                raise RuntimeError(f"STE100Guard baslatilamadi: {error_msg}") from e
                
        self.nlp = _SPACY_MODEL

    def _load_json(self, path: str, key: str = None) -> List[Dict[str, Any]]:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if key and isinstance(data, dict) and key in data:
                        return data.get(key, [])
                    return data if isinstance(data, list) else []
            except Exception as e:
                logger.error("JSON yukleme hatasi (%s): %s", path, e, exc_info=True)
        else:
            logger.warning("Kural dosyasi bulunamadi: %s", path)
        return []

    def _load_json_list(self, path: str, key: str) -> List[str]:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and key in data:
                        return data.get(key, [])
                    return data if isinstance(data, list) else []
            except Exception as e:
                logger.error("JSON yukleme hatasi (%s): %s", path, e, exc_info=True)
        else:
            logger.warning("Core kural dosyasi bulunamadi: %s", path)
        return []

    def build_injection_prompt(self, context_text: str, template_type: str = "General") -> str:
        prompt_parts = []

        prompt_parts.append("<CORE_RULES>")
        for rule in self.core_rules:
            prompt_parts.append(f"- {rule}")
        prompt_parts.append("</CORE_RULES>\n")

        prompt_parts.append("<DYNAMIC_RULES>")
        prompt_parts.append(f"Output Format Required: {template_type.upper()}")
        
        # ENTEGRE EDILEN 2. YONTEM: Olmayan dosyayi beklemeden dogrudan mevcut kurallari uygula
        if template_type.lower() == "procedure":
            prompt_parts.append("- Write short sentences. Use a maximum of 20 words in each sentence.")
            prompt_parts.append("- Write only one instruction in each sentence unless two or more actions occur at the same time.")
            prompt_parts.append("- Write instructions in the imperative(command) form.")
            prompt_parts.append("- If you start an instruction with a descriptive statement(dependent phrase or clause), divide that statement from the command with a comma.")
            prompt_parts.append("- Write notes only to give information, not instructions.")
        elif template_type.lower() == "descriptive":
            prompt_parts.append("- Give information gradually.")
            prompt_parts.append("- Use key words and phrases to organize your text logically.")
            prompt_parts.append("- Write short sentences. Use a maximum of 25 words in each sentence.")
            prompt_parts.append("- Use paragraphs to show related information.")
            prompt_parts.append("- Make sure that each paragraph has only one topic.")
            prompt_parts.append("- Make sure that no paragraph has more than six sentences.")
        elif template_type.lower() == "safety":
            prompt_parts.append("- Use an applicable word(e.g.'warning' or 'caution') to identify the level of risk.")
            prompt_parts.append("- Start a safety instruction with a clear and simple command or condition.")
            prompt_parts.append("- Give an explanation to show the specific risk or possible result.")
                
        prompt_parts.append("</DYNAMIC_RULES>\n")

        _, lexicon_restrictions = self.analyze_and_report(context_text)
        if lexicon_restrictions:
            prompt_parts.append("<DICTIONARY_RESTRICTIONS>")
            prompt_parts.append("Based on the provided context, apply these specific word replacements:")
            prompt_parts.extend(lexicon_restrictions)
            prompt_parts.append("</DICTIONARY_RESTRICTIONS>")

        return "\n".join(prompt_parts)

    def analyze_and_report(self, text: str) -> Tuple[bool, List[str]]:
        if not hasattr(self, 'nlp') or self.nlp is None:
            error_msg = "Sistem Hatasi: NLP modeli yuklenmedigi icin analiz yapilamiyor."
            logger.error(error_msg)
            return False, [error_msg]
            
        if not self.dictionary_rules or not text:
            return True, []

        doc = self.nlp(text)
        feedback_report = []
        
        rule_map = {}
        for rule in self.dictionary_rules:
            kw = rule.get("keyword", "").lower()
            if kw not in rule_map:
                rule_map[kw] = []
            rule_map[kw].append(rule)

        for token in doc:
            if token.is_punct or token.is_space:
                continue

            lemma = token.lemma_.lower()
            text_lower = token.text.lower()
            pos = token.pos_
            
            rules_to_check = rule_map.get(lemma, []) + rule_map.get(text_lower, [])
            
            matched_rules = []
            for rule in rules_to_check:
                if rule not in matched_rules:
                    matched_rules.append(rule)

            for rule in matched_rules:
                is_approved = rule.get("is_approved", True)
                if is_approved:
                    continue
                    
                target_pos = rule.get("pos", "")
                
                if target_pos:
                    if isinstance(target_pos, str) and pos != target_pos.upper():
                        continue
                    elif isinstance(target_pos, list) and pos not in [p.upper() for p in target_pos]:
                        continue
                        
                alternatives = rule.get("approved_alternatives", [])
                alts_str = ", ".join(alternatives) if alternatives else "an approved alternative"
                
                pos_desc = pos if pos else "word"
                msg = f"- DO NOT USE the {pos_desc} '{token.text}'. Instead, use: {alts_str}."
                
                if msg not in feedback_report:
                    feedback_report.append(msg)

        is_compliant = len(feedback_report) == 0
        return is_compliant, feedback_report