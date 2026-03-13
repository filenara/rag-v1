import os
import json
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class STE100Injector:
    def __init__(
        self, 
        dictionary_path: str = "config/ste100_rules.json", 
        structural_rules_path: str = "config/ste100_structural_rules.json"
    ):
        self.dictionary_path = dictionary_path
        self.structural_rules_path = structural_rules_path
        
        self.dictionary_rules = self._load_json(self.dictionary_path)
        self.structural_rules = self._load_json(self.structural_rules_path)
        
        self.core_rules = [
            "You can use words that are: approved in the dictonary, techincal names, techincal verbs.(e.g., 'use' is an approved word in the dictionary, 'engine' is a techinal name, 'ream' is a technical verb.)",
            "Use approved words from the dictionary only as the part of speech given. (e.g., The word 'dim' is an approved adjective, but not an approved verb. non-STE: Dim the lights. STE: The lights in the cabin become dim.)",
            "Use approved word only with their approved meanings.(e.g., The  approved meaning of the word 'follow' is 'come after' and not 'obey'. Non-STE: Follow the instructions. STE: Obey the safety instructions.)",
            "Use only the approved forms of verbs and adjectives. (e.g., REMOVE(v), REMOVES, REMOVED, REMOVED. This word tells you that you can use the approved verb 'remove' as infinitve, present tense, simple past tense, past participle.)",
            "You can use words that you can include in a technical name category. Words are technical name if you can include them in one of these 19 categories: 1.Names in the official parts information(e.g., bolt, cable, clip), 2.Names of vehicles or machine and locations on them(e.g.,aircraft, engine room), 3.Names of tools and support equipment, their parts and locations on them(e.g.,rigging pin, brush), 4.Names of materials, consumables, and unwanted material(e.g.,compound, sealant), 5.Name of facilities, infrastructure, their parts, and locations(e.g., dock, engine shop floor), 6.Name of systems, components and circuits, their functions, configurations, and parts(e.g.,air conditioning, power unit), 7.Mathematical, scientific, and engineering terms(e.g.,acceleration, idle speed, radius), 8.Navigation and geographic terms(e.g.,air, Turkey, west), 9.Numbers, units of measurement and time(e.g., 28, second,meter'm'), 10.Quoted text such as that on placards, labels, signs, markings, and display units(e.g.,abort button, INOP system), 11.Names of persons, groups, pr organizations(e.g.,air traffic control, operator), 12.Parts of body(e.g.,ear, eyes), 13.Common personel effects(e.g.,Cigarette lighter, clothing), 14.Medical terms(e.g.,allergy,asthma), 15.Names of official documents and parts of documentation(e.g.,Checklist, Class), 16.Enviromental and operational conditions(e.g.,cloud, humidity), 17.Colors(e.g.,beige,cyan blue), 18.Damage terms(e.g.,buckle, chafing), 19.Information technology and telephony terms(e.g.,HTML, backup). This are only example not the complete list of all possible techinical names.",
            "Use a word that is unapproved in the dictionary only when it is a technical name or part of a technical name.(e.g.,'Base' is an unapproved word in the dictionary. But you can use this word as a technical name. STE:The base of triangle is 5cm. Non-STE: Make sure that the two spigots at the base of the unit engage.)",
            "Do not use words that are technical names as verbs.(e.g.,'Oil' is a technical name. Do not use 'oil' as a verb.)"
            "Use technical names that agree with approved nomenclature. If there is a designated technical name for a system, component, part, or process, use that technical name.(e.g.,STE:The front panel of the phone has a touchscreen and a home button.)",
            "When you must select a technical name, use one which is short and easy to understand.(e.g., Non-STE: Remove the four stainless steel pan headmachine screws(10), that attach the metallic machined flange(15) to the front housing cover(20). STE:Remove the four screw(10) that attach the flange(15) to the cover(20).).",
            "Do not use slang or jargon word as technical names. (e.g., Non-STE:Make a sandwich with two washers and the spacer. STE:Install the spacer between the two washers.)",
            "Do not use different technical names for the same item. You must referred to same item with same technical name.",
            "You can use verbs that you can include in a technical verb category. Words are technical verbs if you can include them in one of these four categories: 1.Manufacturing process(e.g.,drill,flame,bond,braze, grind, anneal, buff, extrude) 2.Computer processes and applications(e.g.,click, enter, clear, delete, debug), 3.Descriptions(e.g.,bisect, aim, arm, detect, comply with, load), 4.Operational language(e.g.,airdrop, alert, brief, fly, rotate). This are only example not the complete list of all possible techinical names.",
            "Do not use technical verbs as nouns. (e.g.,Non-STE:Give the hole 0.20-inch over ream. STE:Ream the hole 0.20 inch larger than the standard.)",
            "Use American English spelling.(e.g.,Non-STE:Change the colour of the display. STE:Cahnge the color of the display.)",
            "Write noun clusters of no more than three words.(e.g.,Horizontal cylinder pivot bearing)",
            "When a technical name has more than three words, write it in full. Then you can simplify it as follows: -Give a shorter name or -Use hypens (-) between words that are used as a single unit.(e.g., STE: Before you do this procedure, engage the ramp service door safety connector pin(the pin that holds the ramp service door, referred to in this procedure as the 'safety connector pin'). STE:Make sure that the landing-light cutoff-switch power connection is safe.",
            "When applicable, use an article (the, a, an) or a demonstrative adjective (this, these) before a noun. (e.g.,Non-STE:Turn shaft assembly. STE:Turn the shaft assembly."
            "Use only those forms of the verb that are given in the dictionary.",
            "Use the approved forms of the verb to make only: -The infinitive, -The imperative(command form), -The simple present tense, -The simple past tense, -The past participle(as an adjective), -The future tense"
            "Use the past participle only as an adjective.(e.g.,STE:Connect the disconnected wires.('Disconnected' is an adjective before the noun 'wires'.),"
            "Do not use helping verbs to make complex verb structures.(e.g.,Non-STE:The operator has adjusted the linkage. STE:The operator adjust the linkage."
            "Use the '-ing' form of a verb only as a modifier in the technical name.They can be part of a verb to describe an action in the present, adjectives, nouns or parts of noun phrases; long strings of modifiers, noun phrases, and dependent clauses; because these different functions in a sentence can often couse ambiguity or lead to long, complex sentences, it is generally not permitted in STE to use words that end in '-ing'"

Sayfa 42de kalındı
        ]

    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "rules" in data:
                        return data.get("rules", [])
                    return data if isinstance(data, list) else []
            except Exception as e:
                logger.error(f"JSON yukleme hatasi ({path}): {e}")
        else:
            logger.warning(f"Kural dosyasi bulunamadi: {path}")
        return []

    def _extract_forbidden_words(self, context_text: str) -> List[str]:
        if not self.dictionary_rules or not context_text:
            return []

        words_in_context = set(re.findall(r'\b[a-zA-Z]+\b', context_text.lower()))
        injected_lexicon = []

        for rule in self.dictionary_rules:
            keyword = rule.get("keyword", "").lower()
            is_approved = rule.get("is_approved", True)
            
            if keyword in words_in_context and not is_approved:
                alternatives = rule.get("approved_alternatives", [])
                alts_str = ", ".join(alternatives) if alternatives else "an approved alternative"
                injected_lexicon.append(f"- DO NOT USE '{keyword}'. Instead, use: {alts_str}")

        return injected_lexicon

    def build_injection_prompt(self, context_text: str, template_type: str = "General") -> str:
        prompt_parts = []

        prompt_parts.append("<CORE_RULES>")
        for rule in self.core_rules:
            prompt_parts.append(f"- {rule}")
        prompt_parts.append("</CORE_RULES>\n")

        prompt_parts.append("<DYNAMIC_RULES>")
        prompt_parts.append(f"Output Format Required: {template_type.upper()}")
        
        has_structural_rules = False
        if self.structural_rules:
            for rule in self.structural_rules:
                if rule.get("category", "") in [template_type, "General"]:
                    prompt_parts.append(f"- {rule.get('rule')}")
                    has_structural_rules = True
                    
        if not has_structural_rules:
            if template_type.lower() == "procedure":
                prompt_parts.append("- Write as a numbered list (1., 2., 3.).")
                prompt_parts.append("- Give only one instruction per step.")
                prompt_parts.append("- Use the active voice and imperative mood. Do NOT use passive voice.")
            elif template_type.lower() == "descriptive":
                prompt_parts.append("- Write in short, descriptive paragraphs.")
                prompt_parts.append("- State facts clearly. Do not give commands or instructions.")
            elif template_type.lower() == "preliminary":
                prompt_parts.append("- Use bullet points to list required tools, parts, or safety warnings.")
                
        prompt_parts.append("</DYNAMIC_RULES>\n")

        lexicon_restrictions = self._extract_forbidden_words(context_text)
        if lexicon_restrictions:
            prompt_parts.append("<DICTIONARY_RESTRICTIONS>")
            prompt_parts.append("Based on the provided context, apply these specific word replacements:")
            prompt_parts.extend(lexicon_restrictions)
            prompt_parts.append("</DICTIONARY_RESTRICTIONS>")

        return "\n".join(prompt_parts)