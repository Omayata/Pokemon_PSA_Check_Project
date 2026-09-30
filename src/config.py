"""โหลดค่าตั้งจาก configs/params.yaml (แหล่งความจริงเดียวของทั้งระบบ)"""

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS = ROOT / "configs" / "params.yaml"


@lru_cache
def load_params(path: str | None = None) -> dict:
    params_path = Path(path or os.getenv("PARAMS_PATH", DEFAULT_PARAMS))
    with open(params_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_name(name: str) -> str:
    """'Corner-Wear' / 'corner_wear' / 'Corner Wear' -> 'corner wear'"""
    return " ".join(name.replace("-", " ").replace("_", " ").lower().split())
