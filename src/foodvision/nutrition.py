"""Approximate per-serving nutrition lookup for each dish class.

Calories are looked up from a table after classification, not estimated from
the photo: portion size can't be judged reliably from a single image.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

NUTRITION_PATH = Path(__file__).with_name("nutrition.json")
DISCLAIMER = ("Approximate values for one typical serving; real values vary by recipe "
              "and portion. Not dietary advice.")


@lru_cache(maxsize=1)
def load_table() -> dict[str, dict]:
    table = json.loads(NUTRITION_PATH.read_text())
    table.pop("_meta", None)
    return table


def lookup(dish: str) -> dict | None:
    return load_table().get(dish)
