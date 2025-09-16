import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import requests
from tqdm import tqdm


def _safe_filename(url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return f"{h}.jpg"


def download_image(url: str, out_dir: str, timeout: int = 20) -> Tuple[str, bool]:
    os.makedirs(out_dir, exist_ok=True)
    fname = _safe_filename(url)
    path = os.path.join(out_dir, fname)
    if os.path.exists(path):
        return path, True
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}, stream=True)
        if resp.status_code != 200:
            return path, False
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return path, True
    except Exception:
        return path, False


def download_catalog_images(products_jsonl: str, out_dir: str, max_workers: int = 16, top_n_per_product: int = 2) -> str:
    manifest_path = os.path.join(out_dir, "images_manifest.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    # Load products
    products: List[Dict] = []
    with open(products_jsonl, "r") as f:
        for line in f:
            products.append(json.loads(line))

    tasks: List[Tuple[str, str]] = []
    for p in products:
        for url in (p.get("image_urls") or [])[:top_n_per_product]:
            tasks.append((p["id"], url))

    results: List[Tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(download_image, url, out_dir): (pid, url) for pid, url in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
            pid, url = futures[fut]
            path, ok = fut.result()
            if ok:
                results.append((pid, path))

    # Write manifest
    with open(manifest_path, "w") as f:
        for pid, path in results:
            f.write(json.dumps({"id": pid, "image_path": path}) + "\n")

    return manifest_path
