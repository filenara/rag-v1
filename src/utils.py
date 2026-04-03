import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Ayar dosyasi okuma hatasi ({config_path}): {e}")
    else:
        logger.warning(f"Ayar dosyasi ({config_path}) bulunamadi.")
    return {}


def load_secrets(secrets_path: str = "config/secrets.yaml") -> Dict[str, Any]:
    path = Path(secrets_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Gizli ayar dosyasi okuma hatasi ({secrets_path}): {e}")
    else:
        logger.warning(f"Gizli ayar dosyasi ({secrets_path}) bulunamadi.")
    return {}


def load_prompts() -> Dict[str, str]:
    prompts_path = Path("config/prompts.yaml")
    
    if prompts_path.exists():
        try:
            with open(prompts_path, "r", encoding="utf-8") as f:
                loaded_prompts = yaml.safe_load(f)
                if isinstance(loaded_prompts, dict):
                    return loaded_prompts
                logger.warning("Prompts dosyasi gecerli bir sozluk degil.")
        except Exception as e:
            logger.error(f"Hata: {prompts_path} okunamadi: {e}")
    else:
        logger.warning(f"{prompts_path} bulunamadi. Varsayilan promptsuz islem yapilacak.")
        
    return {}