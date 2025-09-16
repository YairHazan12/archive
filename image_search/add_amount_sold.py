import json
import os
import random
import hashlib
from typing import List, Dict

META_PATH = "/Users/yairhazan/Downloads/archive/vector_index/image_meta.json"


def deterministic_amount_sold(item_id: str) -> int:
    # Deterministic pseudo-random in [0, 2000] based on item id hash
    h = hashlib.sha256(item_id.encode("utf-8")).hexdigest()
    n = int(h[:8], 16)
    return n % 2001


def add_amount_sold(meta_path: str = META_PATH) -> None:
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)
    with open(meta_path, "r") as f:
        meta: List[Dict] = json.load(f)
    changed = False
    for rec in meta:
        if "amount_sold" not in rec:
            rec["amount_sold"] = deterministic_amount_sold(rec.get("id", ""))
            changed = True
    if changed:
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        print(f"Updated {meta_path} with amount_sold")
    else:
        print("No changes; amount_sold already present")


if __name__ == "__main__":
    add_amount_sold()
