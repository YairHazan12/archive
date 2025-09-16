import os
import argparse

from image_search.csv_loader import load_products, save_products_jsonl
from image_search.downloader import download_catalog_images
from image_search.build_index import build_index


def main():
    parser = argparse.ArgumentParser(description="Build image similarity index from CSVs")
    parser.add_argument("--root", default="/Users/yairhazan/Downloads/archive")
    parser.add_argument("--work_dir", default="/Users/yairhazan/Downloads/archive/.work")
    parser.add_argument("--images_dir", default="/Users/yairhazan/Downloads/archive/images_cache")
    parser.add_argument("--index_dir", default="/Users/yairhazan/Downloads/archive/vector_index")
    parser.add_argument("--model", default="clip-ViT-B-32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha_image", type=float, default=0.7, help="Weight for image in fusion; text gets 1-alpha")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    products = load_products(args.root)
    products_path = os.path.join(args.work_dir, "products.jsonl")
    save_products_jsonl(products, products_path)
    print(f"Saved {len(products)} products -> {products_path}")

    manifest = download_catalog_images(products_path, args.images_dir)
    print(f"Manifest at {manifest}")

    index_path = build_index(
        manifest,
        args.index_dir,
        model_name=args.model,
        device=args.device,
        products_jsonl=products_path,
        alpha_image=args.alpha_image,
    )
    print(f"Index built at {index_path}")


if __name__ == "__main__":
    main()
