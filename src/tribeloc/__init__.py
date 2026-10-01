"""TRIBE v2 localization of the 46 LLM-modularity tasks."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    with open(path or ROOT / "config" / "analysis.yaml") as f:
        return yaml.safe_load(f)
