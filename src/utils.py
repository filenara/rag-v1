import yaml
import json
import os


def load_config(config_path="config/settings.yaml"):
    """Sistem ayarlarini barindiran settings.yaml dosyasini yukler."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Ayar dosyasi okuma hatasi ({config_path}): {e}")
    else:
        print(f"Uyari: Ayar dosyasi ({config_path}) bulunamadi.")
    return {}


def load_secrets(secrets_path="config/secrets.yaml"):
    """Kimlik dogrulama ayarlarini barindiran secrets.yaml dosyasini yukler."""
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Gizli ayar dosyasi okuma hatasi ({secrets_path}): {e}")
    else:
        print(f"Uyari: Gizli ayar dosyasi ({secrets_path}) bulunamadi.")
    return {}


def load_prompts():
    """
    Tum prompt metinlerini (prompts.yaml) yukler.
    Boylece kod icinde uzun stringler tutmak zorunda kalmayiz.
    """
    prompts_path = "config/prompts.yaml"
    
    # Varsayilan Promptlar (Dosya yoksa hata vermesin diye)
    default_prompts = {
        "system_persona": "You are a helpful assistant.",
        "ste100_rules": "",
        "response_template": (
            "{intent_instruction}\n\n"
            "[CONVERSATION HISTORY]\n{history_text}\n\n"
            "[RETRIEVED CONTEXT]\n{context_text}\n\n"
            "[USER QUESTION]\n{query}"
        )
    }

    if os.path.exists(prompts_path):
        try:
            with open(prompts_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Hata: {prompts_path} okunamadi: {e}")
            return default_prompts
    else:
        print(f"Uyari: {prompts_path} bulunamadi. Varsayilanlar kullaniliyor.")
        return default_prompts

