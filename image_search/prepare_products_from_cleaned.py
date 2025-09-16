import csv
import json
import os
from typing import Generator, Tuple


def iter_cleaned_rows(cleaned_root: str) -> Generator[Tuple[str, str, str, str, str], None, None]:
    for gender in ("Men", "Women"):
        gender_dir = os.path.join(cleaned_root, gender)
        if not os.path.isdir(gender_dir):
            continue
        for fname in os.listdir(gender_dir):
            if not fname.lower().endswith(".csv"):
                continue
            category = fname[:-4]
            path = os.path.join(gender_dir, fname)
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield (
                        row.get("id", ""),
                        row.get("name", "") or "",
                        row.get("details", "") or "",
                        row.get("gender", gender),
                        row.get("category", category),
                    )


def write_products_jsonl(cleaned_root: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as out:
        for pid, name, details, gender, category in iter_cleaned_rows(cleaned_root):
            if not pid:
                continue
            rec = {
                "id": pid,
                "name": name,
                "details": details,
                "gender": gender,
                "category": category,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    cleaned_root = os.path.join(repo_root, "cleaned_csv")
    out_path = os.path.join(repo_root, ".work", "products_cleaned.jsonl")
    write_products_jsonl(cleaned_root, out_path)
    print(out_path)


