import os
# Constrain thread counts to avoid segfaults on macOS/Python 3.13
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
from typing import List

import faiss  # type: ignore
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


def load_index(index_dir: str):
    index_path = os.path.join(index_dir, "image_index.faiss")
    meta_path = os.path.join(index_dir, "image_meta.json")
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        raise FileNotFoundError("Index or metadata not found. Build the index first.")
    index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return index, meta


def search_image(query_image_path: str, index_dir: str, top_k: int = 5, model_name: str = "clip-ViT-B-32", device: str = "cpu") -> List[dict]:
    if not os.path.isfile(query_image_path):
        raise FileNotFoundError(f"Query image not found: {query_image_path}")
    index, meta = load_index(index_dir)
    model = SentenceTransformer(model_name, device=device)

    img = Image.open(query_image_path).convert("RGB")
    q = model.encode([img], convert_to_numpy=True, normalize_embeddings=True, num_workers=0, show_progress_bar=False).astype("float32")
    scores, idxs = index.search(q, top_k)
    results: List[dict] = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        item = meta[idx]
        results.append({
            "score": float(score),
            **item,
        })
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--index_dir", default="vector_index")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--model", default="clip-ViT-B-32")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    hits = search_image(args.image, args.index_dir, top_k=args.top_k, model_name=args.model, device=args.device)
    for h in hits:
        print(json.dumps(h, ensure_ascii=False))
