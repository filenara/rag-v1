import yaml
import json
import os

def load_config():
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_secrets():
    with open("config/secrets.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_ste100_rules():
    with open("config/ste100_rules.json", "r", encoding="utf-8") as f:
        return json.load(f)