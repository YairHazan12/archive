import csv
import os
import re
from typing import List, Tuple

import pandas as pd

from .csv_loader import list_category_csvs


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    value = text.strip()
    if not value or value.lower() == "nan":
        return ""
    # Remove URLs
    value = re.sub(r"https?://\S+", "", value)
    # Remove boilerplate endings often present in scraped descriptions
    value = re.sub(r"\bView more\b.*$", "", value, flags=re.IGNORECASE)
    # Remove measurement tails like Height x Length x Width ... cm / ...″
    value = re.sub(r"Height\s*x\s*Length(?:\s*x\s*Width)?[^\n]*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"Length\s*of\s*inner\s*leg\s*seam:[^\n]*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"Length\s*of\s*outer\s*leg\s*seam:[^\n]*", "", value, flags=re.IGNORECASE)
    # Remove unit-heavy parentheticals and sizes
    value = re.sub(r"\bcm\b|\binches\b|\bin\.|\bmm\b|\bml\b|\box\b|\bLitre\b|\blitre\b", "", value)
    # Collapse extra whitespace
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _normalize_header(df: pd.DataFrame) -> Tuple[str, str, str, str, str]:
    name_col = next((c for c in df.columns if c.strip().lower() in {"product_name", "name"}), None)
    link_col = next((c for c in df.columns if c.strip().lower() in {"link", "product_link"}), None)
    img_col = next((c for c in df.columns if c.strip().lower() in {"product_image", "product_images", "image", "images"}), None)
    price_col = next((c for c in df.columns if c.strip().lower() in {"price"}), None)
    details_col = next((c for c in df.columns if c.strip().lower() in {"details", "description"}), None)
    return name_col or "", link_col or "", img_col or "", price_col or "", details_col or ""


def clean_all_to_csv(root_dir: str, out_root: str) -> None:
    for gender, category, path in list_category_csvs(root_dir):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        name_col, link_col, img_col, price_col, details_col = _normalize_header(df)
        if not name_col or not link_col:
            continue

        out_dir = os.path.join(out_root, gender)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{category}.csv")

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "details", "gender", "category"])

            for _, row in df.iterrows():
                try:
                    rid = str(row.get(df.columns[0], "")).strip()
                    link = str(row.get(link_col, "")).strip()
                    if not rid or rid == "nan":
                        rid = link.rsplit("/", 1)[-1] if link else ""
                    if not link:
                        continue
                    name_raw = str(row.get(name_col, "")).strip()
                    name = re.sub(r"https?://\S+", "", name_raw).strip()
                    details_raw = "" if not details_col else str(row.get(details_col, "")).strip()
                    details = _clean_text(details_raw)

                    writer.writerow(
                        [
                            f"{gender}:{category}:{rid}",
                            name,
                            details,
                            gender,
                            category,
                        ]
                    )
                except Exception:
                    continue


if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_root = repo_root
    out_root = os.path.join(repo_root, "cleaned_csv")
    clean_all_to_csv(data_root, out_root)


