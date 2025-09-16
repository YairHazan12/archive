import ast
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup


@dataclass
class ProductRecord:
    id: str
    name: str
    link: str
    image_urls: List[str]
    price: Optional[str]
    details: Optional[str]
    gender: str
    category: str


def list_category_csvs(root_dir: str) -> List[Tuple[str, str, str]]:
    records: List[Tuple[str, str, str]] = []
    for gender in ["Men", "Women"]:
        gender_dir = os.path.join(root_dir, gender, gender)
        if not os.path.isdir(gender_dir):
            continue
        for fname in os.listdir(gender_dir):
            if not fname.lower().endswith(".csv"):
                continue
            records.append((gender, fname[:-4], os.path.join(gender_dir, fname)))
    return records


def parse_image_list(cell: str) -> List[str]:
    if not isinstance(cell, str):
        return []
    text = cell.strip()
    if not text or text == "[]":
        return []
    try:
        value = ast.literal_eval(text)
        urls: List[str] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for k in item.keys():
                        if isinstance(k, str) and k.startswith("http"):
                            urls.append(k)
        return urls
    except Exception:
        return []


def scrape_primary_image(product_url: str, timeout: int = 10) -> List[str]:
    try:
        resp = requests.get(product_url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "lxml")
        # Try OpenGraph image
        og = soup.find("meta", attrs={"property": "og:image"})
        if og and og.get("content"):
            return [og["content"]]
        # Try any image with plausible Zara CDN
        imgs = [img.get("src") for img in soup.find_all("img") if img.get("src")]
        imgs = [u for u in imgs if u.startswith("http")]
        return imgs[:1] if imgs else []
    except Exception:
        return []


def load_products(root_dir: str) -> List[ProductRecord]:
    products: List[ProductRecord] = []
    for gender, category, path in list_category_csvs(root_dir):
        try:
            # Some files have slight header differences; let pandas infer
            df = pd.read_csv(path)
        except Exception:
            continue
        # Normalize column names
        cols = {c.strip().lower().replace(" ", "_") for c in df.columns}
        name_col = next((c for c in df.columns if c.strip().lower() in {"product_name", "name"}), None)
        link_col = next((c for c in df.columns if c.strip().lower() in {"link", "product_link"}), None)
        img_col = next((c for c in df.columns if c.strip().lower() in {"product_image", "product_images", "image", "images"}), None)
        price_col = next((c for c in df.columns if c.strip().lower() in {"price"}), None)
        details_col = next((c for c in df.columns if c.strip().lower() in {"details", "description"}), None)

        if link_col is None or name_col is None:
            continue

        for _, row in df.iterrows():
            try:
                # ID: try first column if numeric index; else derive from link
                rid = str(row.get(df.columns[0], "")).strip()
                if not rid or rid == "nan":
                    rid = str(row.get("id", "")).strip()
                link = str(row.get(link_col, "")).strip()
                if not rid:
                    rid = link.rsplit("/", 1)[-1] if link else os.urandom(4).hex()
                name = str(row.get(name_col, "")).strip()
                price = None if price_col is None else str(row.get(price_col, "")).strip()
                details = None if details_col is None else str(row.get(details_col, "")).strip()

                image_urls: List[str] = []
                if img_col is not None:
                    image_urls = parse_image_list(str(row.get(img_col, "")).strip())
                if not image_urls and link:
                    # Fallback: try scraping one image
                    image_urls = scrape_primary_image(link)

                if not image_urls:
                    # Skip products without any image candidates
                    continue

                products.append(
                    ProductRecord(
                        id=f"{gender}:{category}:{rid}",
                        name=name,
                        link=link,
                        image_urls=image_urls,
                        price=price if price and price.lower() != "nan" else None,
                        details=details if details and details.lower() != "nan" else None,
                        gender=gender,
                        category=category,
                    )
                )
            except Exception:
                continue
    return products


def save_products_jsonl(products: List[ProductRecord], out_path: str) -> None:
    with open(out_path, "w") as f:
        for p in products:
            f.write(json.dumps(p.__dict__, ensure_ascii=False) + "\n")
