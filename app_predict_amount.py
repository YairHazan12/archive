import os
import json
import tempfile
import base64
from io import BytesIO
from glob import glob
from typing import List
import zipfile

import streamlit as st
from PIL import Image

from image_search.query import search_image
from image_search.query_avg import search_topk, average_amount_sold

INDEX_DIR = "/Users/yairhazan/Downloads/archive/vector_index_cleaned"
MODEL = "clip-ViT-B-32"
DEVICE = "cpu"

st.set_page_config(page_title="Predict Amount", page_icon="🛍️", layout="wide")

st.title("Predict Amount")
st.caption("Upload an image to find similar items, estimate average amount sold, or generate a folder report.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
    mode = st.radio("Mode", options=["Similar Items", "Average Amount Sold", "Batch Folder Report"], index=0)

uploaded = None
batch_files: List = []
if mode == "Batch Folder Report":
    batch_files = st.file_uploader(
        "Select images or upload a .zip with images",
        type=["jpg", "jpeg", "png", "webp", "zip"],
        accept_multiple_files=True,
    )
else:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

col1, col2 = st.columns([1, 2])

with col1:
    if uploaded is not None and mode != "Batch Folder Report":
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Query Image", use_column_width=True)

primary_btn_label = "Generate Report" if mode == "Batch Folder Report" else "Run"
run = st.button(primary_btn_label)


def _img_to_data_uri(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _path_img_to_data_uri(path: str) -> str:
    try:
        with Image.open(path).convert("RGB") as im:
            return _img_to_data_uri(im)
    except Exception:
        return ""

if run and mode != "Batch Folder Report" and uploaded is not None:
    with st.spinner("Searching..."):
        # Save to temp file for existing pipeline functions
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            Image.open(uploaded).convert("RGB").save(tmp_path, format="JPEG")
        try:
            if mode == "Similar Items":
                hits = search_image(tmp_path, INDEX_DIR, top_k=top_k, model_name=MODEL, device=DEVICE)
                with col2:
                    st.subheader("Similar Items")
                    for h in hits:
                        cols = st.columns([1, 3])
                        with cols[0]:
                            st.image(h.get("image_path"), use_column_width=True)
                        with cols[1]:
                            st.markdown(f"**Score:** {h.get('score'):.3f}")
                            title = h.get("name") or h.get("id")
                            if h.get("link"):
                                st.markdown(f"**Item:** [{title}]({h.get('link')})")
                            else:
                                st.markdown(f"**Item:** {title}")
                            if h.get("price"):
                                st.markdown(f"**Price:** {h.get('price')}")
                            if h.get("details"):
                                st.caption(h.get("details"))
            else:
                hits = search_topk(tmp_path, INDEX_DIR, top_k=top_k, model_name=MODEL, device=DEVICE)
                avg = average_amount_sold(hits)
                with col2:
                    st.subheader("Average Amount Sold")
                    st.metric(label=f"Average over top {top_k}", value=f"{avg:.1f}")
                    st.divider()
                    st.subheader("Top Matches")
                    for h in hits:
                        cols = st.columns([1, 3])
                        with cols[0]:
                            st.image(h.get("image_path"), use_column_width=True)
                        with cols[1]:
                            st.markdown(f"**Score:** {h.get('score'):.3f}")
                            title = h.get("name") or h.get("id")
                            if h.get("link"):
                                st.markdown(f"**Item:** [{title}]({h.get('link')})")
                            else:
                                st.markdown(f"**Item:** {title}")
                            st.markdown(f"**Amount sold:** {h.get('amount_sold', 'N/A')}")
                            if h.get("price"):
                                st.markdown(f"**Price:** {h.get('price')}")
                            if h.get("details"):
                                st.caption(h.get("details"))
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
elif run and mode == "Batch Folder Report":
    if not batch_files:
        st.warning("Please select images or a .zip file.")
    else:
        with st.spinner("Generating folder report..."):
            temp_paths: List[str] = []
            extract_dirs: List[str] = []
            try:
                # Save images and expand zips
                for uf in batch_files:
                    name = uf.name.lower()
                    if name.endswith((".jpg", ".jpeg", ".png", ".webp")):
                        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(name)[1], delete=False) as tmp:
                            tmp.write(uf.read())
                            temp_paths.append(tmp.name)
                    elif name.endswith(".zip"):
                        # Persist uploaded zip then extract to a temp dir
                        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as ztmp:
                            ztmp.write(uf.read())
                            zpath = ztmp.name
                        real_dir = tempfile.mkdtemp(prefix="batch_zip_")
                        extract_dirs.append(real_dir)
                        try:
                            with zipfile.ZipFile(zpath) as zf:
                                zf.extractall(real_dir)
                        finally:
                            try:
                                os.remove(zpath)
                            except Exception:
                                pass
                        # Walk extracted dir for images
                        for root, _, files in os.walk(real_dir):
                            for fn in files:
                                if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                                    temp_paths.append(os.path.join(root, fn))

                files = temp_paths
                if not files:
                    st.warning("No images found in the selected files.")
                else:
                    sections: List[str] = []
                    sections.append(
                        """
<!DOCTYPE html>
<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\" />\n<title>Predict Amount - Folder Report</title>\n<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Helvetica,Arial,sans-serif; margin:24px;}
.q{display:flex;gap:16px;align-items:flex-start;margin:24px 0;padding-bottom:8px;border-bottom:1px solid #e5e7eb;}
.qimg{width:160px;height:160px;object-fit:cover;border-radius:8px;border:1px solid #e5e7eb;}
.qmeta{flex:1}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin-top:12px;}
.card{border:1px solid #e5e7eb;border-radius:8px;padding:8px;}
.cimg{width:100%;height:150px;object-fit:cover;border-radius:6px;border:1px solid #f1f5f9;}
.title{font-weight:600;margin:6px 0 4px;}
.sub{color:#475569;font-size:12px;margin:0}
.score{font-size:12px;color:#334155;margin-top:4px}
h1{margin:0 0 4px} h2{margin:0} .small{color:#64748b;font-size:12px}
</style>\n</head>\n<body>\n<h1>Predict Amount - Folder Report</h1>\n<p class=\"small\">Top {top_k} similar items per query image. Generated on this machine.</p>
"""
                    )

                    for i, fpath in enumerate(files, start=1):
                        try:
                            qimg = Image.open(fpath).convert("RGB")
                        except Exception:
                            continue
                        data_uri = _img_to_data_uri(qimg)
                        # Save temp for pipeline
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                            qimg.save(tmp.name, format="JPEG")
                            tmp_query = tmp.name
                        try:
                            hits = search_image(tmp_query, INDEX_DIR, top_k=top_k, model_name=MODEL, device=DEVICE)
                        finally:
                            try:
                                os.remove(tmp_query)
                            except Exception:
                                pass

                        # Build section HTML
                        title = os.path.basename(fpath)
                        html = ["<section class=\"q\">"]
                        html.append(f"<img class=\"qimg\" src=\"{data_uri}\" alt=\"{title}\" />")
                        html.append("<div class=\"qmeta\">")
                        html.append(f"<h2>{title}</h2>")
                        html.append("<div class=\"grid\">")
                        for h in hits:
                            sim_uri = _path_img_to_data_uri(h.get("image_path", ""))
                            item_name = (h.get("name") or h.get("id") or "").replace("<","&lt;").replace(">","&gt;")
                            details = (h.get("details") or "").replace("<","&lt;").replace(">","&gt;")
                            score = h.get("score")
                            html.append("<div class=\"card\">")
                            if sim_uri:
                                html.append(f"<img class=\"cimg\" src=\"{sim_uri}\" alt=\"{item_name}\" />")
                            html.append(f"<div class=\"title\">{item_name}</div>")
                            if details:
                                html.append(f"<p class=\"sub\">{details}</p>")
                            if isinstance(score, (int, float)):
                                html.append(f"<div class=\"score\">Score: {score:.3f}</div>")
                            html.append("</div>")
                        html.append("</div></div></section>")
                        sections.append("\n".join(html))

                    sections.append("</body>\n</html>")
                    report_html = "\n".join(sections)

                    # Offer as download
                    st.success("Report generated.")
                    st.download_button(
                        label="Download HTML Report",
                        data=report_html.encode("utf-8"),
                        file_name="predict_amount_report.html",
                        mime="text/html",
                    )
            finally:
                # Cleanup extracted directories
                for d in extract_dirs:
                    try:
                        for root, dirs, files in os.walk(d, topdown=False):
                            for fn in files:
                                try:
                                    os.remove(os.path.join(root, fn))
                                except Exception:
                                    pass
                            for dd in dirs:
                                try:
                                    os.rmdir(os.path.join(root, dd))
                                except Exception:
                                    pass
                        os.rmdir(d)
                    except Exception:
                        pass
else:
    with col2:
        if mode == "Batch Folder Report":
            st.info("Select multiple images or upload a .zip and click Generate Report.")
        else:
            st.info("Upload an image and click Run.")
