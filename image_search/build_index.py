import os
# Constrain thread counts to avoid segfaults on macOS/Python 3.13
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
from typing import Dict, List, Tuple, Optional

import faiss  # type: ignore
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_manifest(manifest_path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(manifest_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["id"]
            img = rec["image_path"]
            if os.path.exists(img):
                pairs.append((pid, img))
    return pairs


def load_products_map(products_jsonl: Optional[str]) -> Dict[str, dict]:
    if not products_jsonl or not os.path.exists(products_jsonl):
        return {}
    m: Dict[str, dict] = {}
    with open(products_jsonl, "r") as f:
        for line in f:
            p = json.loads(line)
            m[p["id"]] = p
    return m


def _normalize(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        v = v[None, :]
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return (v / norms).astype("float32")


def build_index(
    manifest_path: str,
    out_dir: str,
    model_name: str = "clip-ViT-B-32",
    device: str = "cpu",
    batch_size: int = 8,
    products_jsonl: Optional[str] = None,
    alpha_image: float = 0.7,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    pairs = load_manifest(manifest_path)
    if not pairs:
        raise RuntimeError("No images found in manifest")

    products_map = load_products_map(products_jsonl)

    model = SentenceTransformer(model_name, device=device)

    image_paths = [p for _, p in pairs]
    product_ids = [pid for pid, _ in pairs]

    embeddings_img_list: List[np.ndarray] = []
    batch_images: List[Image.Image] = []

    def flush_batch():
        nonlocal embeddings_img_list, batch_images
        if not batch_images:
            return
        embs = model.encode(
            batch_images,
            batch_size=len(batch_images) if len(batch_images) < batch_size else batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            num_workers=0,
        )
        embeddings_img_list.append(embs)
        batch_images = []

    for path in tqdm(image_paths, desc="Embedding images"):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        batch_images.append(img)
        if len(batch_images) >= batch_size:
            flush_batch()
    flush_batch()

    X_img = np.vstack(embeddings_img_list).astype("float32")

    # Optional text embeddings
    X_fused = X_img
    if products_map:
        texts: List[str] = []
        for pid in product_ids:
            p = products_map.get(pid, {})
            name = (p.get("name") or "").strip()
            details = (p.get("details") or "").strip()
            # Simple concatenation; customize if needed
            text = name
            if details and details.lower() != "nan":
                text = f"{name}. {details}"
            texts.append(text if text else name)
        X_txt = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            num_workers=0,
        ).astype("float32")
        # Weighted fusion: alpha*image + (1-alpha)*text, then normalize
        alpha = float(alpha_image)
        X_fused = _normalize(alpha * X_img + (1.0 - alpha) * X_txt)

    d = X_fused.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X_fused)

    index_path = os.path.join(out_dir, "image_index.faiss")
    faiss.write_index(index, index_path)

    # Save metadata mapping index row -> product info and image path
    meta_path = os.path.join(out_dir, "image_meta.json")
    meta: List[dict] = []
    for i in range(len(image_paths)):
        pid = product_ids[i]
        base = {"id": pid, "image_path": image_paths[i]}
        if pid in products_map:
            p = products_map[pid]
            base.update({
                "name": p.get("name"),
                "link": p.get("link"),
                "price": p.get("price"),
                "details": p.get("details"),
                "gender": p.get("gender"),
                "category": p.get("category"),
            })
        meta.append(base)
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return index_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", default="vector_index")
    parser.add_argument("--model", default="clip-ViT-B-32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--products", default=None)
    parser.add_argument("--alpha_image", type=float, default=0.7)
    args = parser.parse_args()

    path = build_index(
        args.manifest,
        args.out_dir,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        products_jsonl=args.products,
        alpha_image=args.alpha_image,
    )
    print(f"Index written to {path}")
