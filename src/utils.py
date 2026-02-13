import yaml
import json
import os

def load_config():
    """Genel sistem ayarlarını (settings.yaml) yükler."""
    config_path = "config/settings.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Konfigürasyon dosyası bulunamadı: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_secrets():
    """Şifre ve gizli anahtarları (secrets.yaml) yükler."""
    secrets_path = "config/secrets.yaml"
    if not os.path.exists(secrets_path):
        # Dosya yoksa boş bir yapı döndür veya hata fırlat
        print(f"Uyarı: {secrets_path} bulunamadı.")
        return {"credentials": {"usernames": {}}, "cookie": {"name": "", "key": "", "expiry_days": 1}}
        
    with open(secrets_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_ste100_rules():
    """STE100 yasaklı kelime listesini (ste100_rules.json) yükler."""
    rules_path = "config/ste100_rules.json"
    if os.path.exists(rules_path):
        with open(rules_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # Dosya yoksa varsayılan boş kural döndür
        return {"forbidden_words": {}}