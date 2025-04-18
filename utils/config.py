import json
from typing import Tuple


def load_config(filepath: str) -> Tuple[dict, dict, dict]:
    """
    Returns configs for model, training and data
    """
    with open(filepath, "r") as f:
        config = json.load(f)

    return config["model"], config["train"], config["data"]
