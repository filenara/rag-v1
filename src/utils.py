import yaml
import json
import os

# ... (load_config ve load_secrets fonksiyonları aynı kalacak) ...

def load_prompts():
    """
    YENİ: Tüm prompt metinlerini (prompts.yaml) yükler.
    Böylece kod içinde uzun stringler tutmak zorunda kalmayız.
    """
    prompts_path = "config/prompts.yaml"
    
    # Varsayılan Promptlar (Dosya yoksa hata vermesin diye)
    default_prompts = {
        "system_persona": "You are a helpful assistant.",
        "ste100_rules": "",
        "response_template": "Context: {context}\nQuestion: {question}"
    }

    if os.path.exists(prompts_path):
        try:
            with open(prompts_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Hata: {prompts_path} okunamadı: {e}")
            return default_prompts
    else:
        print(f"Uyarı: {prompts_path} bulunamadı. Varsayılanlar kullanılıyor.")
        return default_prompts

def load_ste100_rules():
    # ... (Eski json yükleme fonksiyonu aynen kalıyor, Guard için gerekli) ...
    rules_path = "config/ste100_rules.json"
    if os.path.exists(rules_path):
        with open(rules_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []