from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _CFG = yaml.safe_load(f)


def get_setting(path: str, default=None):
    parts = path.split(".")
    val = _CFG
    for p in parts:
        if isinstance(val, dict) and p in val:
            val = val[p]
        else:
            return default
    return val


def get_active_profile() -> str:
    return get_setting("profile", "rfp") # default is RFP


def get_prompt_template(kind: str) -> str:
    profile = get_active_profile()
    val = get_setting(f"prompts.{profile}.{kind}")
    if not isinstance(val, str):
        raise ValueError(
            f"Prompt '{kind}' for profile '{profile}' not found or not a string in config.yml."
        )
    return val


def get_docs_dir() -> Path:
    profile = get_active_profile()
    # paths.<profile>.docs_dir
    rel = get_setting(f"paths.{profile}.docs_dir")
    if not rel:
        # fallback: base_docs_dir or 'docs'
        rel = get_setting("paths.base_docs_dir", "docs")
    return PROJECT_ROOT / rel


def get_data_dir() -> Path:
    profile = get_active_profile()
    # paths.<profile>.data_dir
    rel = get_setting(f"paths.{profile}.data_dir")
    if not rel:
        rel = get_setting("paths.base_data_dir", "data")
    return PROJECT_ROOT / rel
