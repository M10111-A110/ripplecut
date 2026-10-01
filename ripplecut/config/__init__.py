"""Bundled configuration package for RippleCut."""
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent

def get_bundled_config(name: str) -> Path:
    return CONFIG_DIR / name
