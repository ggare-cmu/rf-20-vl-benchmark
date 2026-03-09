"""
Prediction Visualization UI — Streamlit
Compares two model detection results against ground-truth bounding boxes.

Model 1 (VQA score):
  results/results_data3/final_consolidated_results/rf-20-vl-benchmark/
  results/eccv26/rf100vl_IPT/Qwen3-VL-30B-A3B-Instruct/rf20_IPT_singleclass_rankScore

Model 2 (Baseline):
  results/results_data3/mllm_fsod_outputs/mllm_fsod/
  results_qwen_3_30b_a3b_instruct/results/Qwen3-VL-30B-A3B-Instruct_instructions_parallel_

Usage:
  streamlit run prediction_viz.py
"""

import os
import json
import glob
import io
from pathlib import Path
from collections import defaultdict

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = [
    "aerial-airport", "dentalai", "flir-camera-objects", "gwhd2021",
    "recode-waste", "wildfire-smoke", "x-ray-id", "soda-bottles",
    "wb-prova", "actions", "aquarium-combined", "the-dreidel-project",
    "orionproducts", "trail-camera", "new-defects-in-wood",
    "lacrosse-object-detection", "defect-detection", "all-elements",
    "water-meter", "paper-parts",
]

MODEL1_ROOT = (
    "results/results_data3/final_consolidated_results/rf-20-vl-benchmark/"
    "results/eccv26/rf100vl_IPT/Qwen3-VL-30B-A3B-Instruct/"
    "rf20_IPT_singleclass_rankScore"
)
MODEL1_SCORE_TYPE = "vqa"
MODEL1_LABEL      = "Model 1 — Qwen3-VL IPT (VQA score)"

MODEL2_ROOT = (
    "results/results_data3/mllm_fsod_outputs/mllm_fsod/"
    "results_qwen_3_30b_a3b_instruct/"
    "results/Qwen3-VL-30B-A3B-Instruct_instructions_parallel_"
)
MODEL2_SCORE_TYPE = "baseline"
MODEL2_LABEL      = "Model 2 — Qwen3-VL Parallel Baseline"

DATASET_ROOT = "./datasets/rf100-vl-fsod"

# Colours (R, G, B, A)
GT_COLOR    = (0,   200,  80,  255)   # green  — ground truth
M1_COLOR    = (30,  144, 255,  220)   # blue   — model 1
M2_COLOR    = (255,  80,  40,  220)   # red    — model 2

BOX_WIDTH   = 3
FONT_SIZE   = 14

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _find_predictions(results_root: str, dataset_name: str, score_type: str):
    """Locate the predictions JSON using the same logic as the reference script."""
    if score_type == "baseline":
        p = os.path.join(results_root, f"predictions_{dataset_name}.json")
        if os.path.isfile(p):
            return p
    else:
        subfolder = "rank" if score_type == "model" else score_type
        p = os.path.join(
            results_root, "final_instruction_eval", "predictions",
            subfolder, f"predictions_{dataset_name}_{score_type}.json"
        )
        if os.path.isfile(p):
            return p

    # Fallback: recursive search
    found = glob.glob(
        os.path.join(results_root, "**", f"predictions_{dataset_name}_{score_type}.json"),
        recursive=True,
    )
    if found:
        return found[0]

    # Baseline fallback without score_type suffix
    found2 = glob.glob(
        os.path.join(results_root, "**", f"predictions_{dataset_name}.json"),
        recursive=True,
    )
    return found2[0] if found2 else None


@st.cache_data(show_spinner=False)
def load_dataset_meta(dataset_name: str):
    """Load COCO annotation file for a dataset."""
    base = os.path.join(DATASET_ROOT, dataset_name)
    if not os.path.isdir(base):
        cands = glob.glob(os.path.join(DATASET_ROOT, f"{dataset_name}*"))
        if not cands:
            return None
        base = cands[0]

    ann_path = os.path.join(base, "test", "_annotations.coco.json")
    if not os.path.isfile(ann_path):
        return None

    with open(ann_path, "r") as f:
        data = json.load(f)

    img_dir = os.path.join(base, "test")
    images  = {img["id"]: img for img in data["images"]}
    cats    = {cat["id"]: cat["name"] for cat in data["categories"]}

    anns_by_img: dict[int, list] = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    return {
        "img_dir":     img_dir,
        "images":      images,
        "cats":        cats,
        "anns_by_img": anns_by_img,
    }


@st.cache_data(show_spinner=False)
def load_predictions(results_root: str, dataset_name: str, score_type: str) -> dict[int, list]:
    """Load predictions JSON and index by image_id."""
    path = _find_predictions(results_root, dataset_name, score_type)
    if path is None or not os.path.isfile(path):
        return {}

    with open(path, "r") as f:
        preds = json.load(f)

    by_img: dict[int, list] = defaultdict(list)
    for p in preds:
        if p.get("category_id", -1) != -1:
            by_img[p["image_id"]].append(p)
    return dict(by_img)


def _get_font(size: int = FONT_SIZE):
    """Try to get a truetype font, fall back to default."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", size)
        except Exception:
            return ImageFont.load_default()


def _draw_boxes(
    draw: ImageDraw.Draw,
    boxes: list[dict],
    color: tuple,
    cats: dict,
    score_key: str = "score",
    show_score: bool = True,
    label_prefix: str = "",
):
    """Draw bounding boxes with class label and optional score."""
    font = _get_font(FONT_SIZE)
    for box in boxes:
        x, y, w, h = box["bbox"]
        x1, y1, x2, y2 = x, y, x + w, y + h

        # Outline rectangle
        for t in range(BOX_WIDTH):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color[:3])

        # Label
        cat_name = cats.get(box.get("category_id", -1), "?")
        score    = box.get(score_key, box.get("score", None))
        if show_score and score is not None:
            label = f"{label_prefix}{cat_name} {score:.2f}"
        else:
            label = f"{label_prefix}{cat_name}"

        # Text background
        try:
            bbox_text = draw.textbbox((x1, y1 - FONT_SIZE - 4), label, font=font)
        except AttributeError:
            w_t, h_t = draw.textsize(label, font=font)
            bbox_text = (x1, y1 - h_t - 4, x1 + w_t, y1)

        pad = 2
        draw.rectangle(
            [bbox_text[0] - pad, bbox_text[1] - pad,
             bbox_text[2] + pad, bbox_text[3] + pad],
            fill=(*color[:3], 200),
        )
        draw.text((x1, y1 - FONT_SIZE - 4), label, fill=(255, 255, 255), font=font)


def render_comparison(
    img_path: str,
    gt_anns: list,
    m1_preds: list,
    m2_preds: list,
    cats: dict,
    score_threshold: float = 0.0,
    show_scores: bool = True,
) -> Image.Image:
    """
    Render a 3-panel comparison image:
      [ Ground Truth | Model 1 | Model 2 ]
    """
    base = Image.open(img_path).convert("RGBA")
    W, H = base.size

    def _panel(boxes, color, label_prefix=""):
        img_copy = base.copy()
        overlay  = Image.new("RGBA", img_copy.size, (0, 0, 0, 0))
        draw     = ImageDraw.Draw(overlay)
        filtered = [b for b in boxes
                    if b.get("score", 1.0) >= score_threshold or "bbox" in b and "score" not in b]
        _draw_boxes(draw, filtered, color, cats,
                    show_score=show_scores, label_prefix=label_prefix)
        return Image.alpha_composite(img_copy, overlay).convert("RGB")

    gt_panel = _panel(gt_anns, GT_COLOR)
    m1_panel = _panel(m1_preds, M1_COLOR)
    m2_panel = _panel(m2_preds, M2_COLOR)

    # Header labels
    font_hdr = _get_font(16)
    panel_labels = ["Ground Truth", MODEL1_LABEL, MODEL2_LABEL]
    panel_imgs   = [gt_panel, m1_panel, m2_panel]
    HDR_H        = 30
    panels_with_header = []

    for lbl, pnl in zip(panel_labels, panel_imgs):
        hdr = Image.new("RGB", (W, HDR_H), (30, 30, 30))
        d   = ImageDraw.Draw(hdr)
        d.text((4, 6), lbl, fill=(255, 255, 255), font=font_hdr)
        combined = Image.new("RGB", (W, H + HDR_H))
        combined.paste(hdr, (0, 0))
        combined.paste(pnl, (0, HDR_H))
        panels_with_header.append(combined)

    SEP = 6
    total_w = W * 3 + SEP * 2
    total_h = H + HDR_H
    canvas  = Image.new("RGB", (total_w, total_h), (60, 60, 60))
    for i, pnl in enumerate(panels_with_header):
        canvas.paste(pnl, (i * (W + SEP), 0))

    return canvas


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Detection Prediction Visualizer",
    layout="wide",
    page_icon="🔍",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Detection Viz")
    st.markdown("**Compare two model predictions against ground truth.**")
    st.divider()

    dataset_name = st.selectbox("Dataset", DATASETS)
    st.divider()

    score_threshold = st.slider(
        "Score threshold (filter predictions)",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05,
    )
    show_scores = st.toggle("Show scores on labels", value=True)
    st.divider()

    st.markdown(
        f"<small>🟢 **GT** &nbsp; 🔵 **{MODEL1_LABEL.split('—')[0].strip()}** &nbsp; 🔴 **{MODEL2_LABEL.split('—')[0].strip()}**</small>",
        unsafe_allow_html=True,
    )

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner(f"Loading dataset: {dataset_name}…"):
    meta = load_dataset_meta(dataset_name)

if meta is None:
    st.error(
        f"❌ Dataset `{dataset_name}` not found.\n\n"
        f"Expected path: `{DATASET_ROOT}/{dataset_name}/test/_annotations.coco.json`"
    )
    st.stop()

with st.spinner("Loading predictions…"):
    m1_preds_all = load_predictions(MODEL1_ROOT, dataset_name, MODEL1_SCORE_TYPE)
    m2_preds_all = load_predictions(MODEL2_ROOT, dataset_name, MODEL2_SCORE_TYPE)

# ── Image selector ────────────────────────────────────────────────────────────
img_ids      = sorted(meta["images"].keys())
img_id_strs  = [f"{meta['images'][i]['file_name']}  (id={i})" for i in img_ids]

col_nav1, col_nav2, col_nav3 = st.columns([1, 6, 1])
with col_nav1:
    if st.button("◀ Prev"):
        idx = st.session_state.get("img_idx", 0)
        st.session_state["img_idx"] = max(0, idx - 1)
with col_nav3:
    if st.button("Next ▶"):
        idx = st.session_state.get("img_idx", 0)
        st.session_state["img_idx"] = min(len(img_ids) - 1, idx + 1)
with col_nav2:
    sel_idx = st.selectbox(
        "Select image",
        range(len(img_ids)),
        index=st.session_state.get("img_idx", 0),
        format_func=lambda i: img_id_strs[i],
        key="img_selector",
    )
    st.session_state["img_idx"] = sel_idx

img_id   = img_ids[st.session_state["img_idx"]]
img_info = meta["images"][img_id]
img_path = os.path.join(meta["img_dir"], img_info["file_name"])

# ── Stats row ─────────────────────────────────────────────────────────────────
gt_anns  = meta["anns_by_img"].get(img_id, [])
m1_preds = m1_preds_all.get(img_id, [])
m2_preds = m2_preds_all.get(img_id, [])

m1_filtered = [p for p in m1_preds if p.get("score", 1.0) >= score_threshold]
m2_filtered = [p for p in m2_preds if p.get("score", 1.0) >= score_threshold]

sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("GT boxes",         len(gt_anns))
sc2.metric(f"Model 1 preds (≥{score_threshold:.2f})", len(m1_filtered))
sc3.metric(f"Model 2 preds (≥{score_threshold:.2f})", len(m2_filtered))
sc4.metric("Image",            f"{img_info['width']}×{img_info['height']}")

# ── Render ────────────────────────────────────────────────────────────────────
if not os.path.isfile(img_path):
    st.error(f"Image file not found: `{img_path}`")
    st.stop()

with st.spinner("Rendering…"):
    comparison = render_comparison(
        img_path       = img_path,
        gt_anns        = gt_anns,
        m1_preds       = m1_filtered,
        m2_preds       = m2_filtered,
        cats           = meta["cats"],
        score_threshold= score_threshold,
        show_scores    = show_scores,
    )

st.image(comparison, use_container_width=True, caption=img_info["file_name"])

# ── Download ──────────────────────────────────────────────────────────────────
img_bytes = image_to_bytes(comparison, fmt="PNG")
dl_name   = f"{dataset_name}_{img_info['file_name'].replace('/', '_')}_comparison.png"

st.download_button(
    label     = "⬇️  Download comparison image",
    data      = img_bytes,
    file_name = dl_name,
    mime      = "image/png",
)

# ── Individual panels + annotation table ─────────────────────────────────────
with st.expander("📊 Annotation details", expanded=False):
    tab_gt, tab_m1, tab_m2 = st.tabs(["Ground Truth", MODEL1_LABEL, MODEL2_LABEL])

    def _ann_table(anns, cats, score_key="score"):
        rows = []
        for a in anns:
            x, y, w, h = a["bbox"]
            rows.append({
                "Class":   cats.get(a.get("category_id", -1), "?"),
                "x":       round(x, 1),
                "y":       round(y, 1),
                "w":       round(w, 1),
                "h":       round(h, 1),
                "Score":   round(a.get(score_key, float("nan")), 4) if score_key in a else "—",
            })
        return rows

    with tab_gt:
        rows = _ann_table(gt_anns, meta["cats"])
        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No GT annotations for this image.")

    with tab_m1:
        rows = _ann_table(m1_filtered, meta["cats"])
        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No Model 1 predictions (above threshold) for this image.")

    with tab_m2:
        rows = _ann_table(m2_filtered, meta["cats"])
        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No Model 2 predictions (above threshold) for this image.")

# ── Individual panel downloads ────────────────────────────────────────────────
with st.expander("⬇️  Download individual panels", expanded=False):
    base_img = Image.open(img_path).convert("RGBA")
    W2, H2   = base_img.size

    def _single_panel_bytes(boxes, color):
        overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        draw    = ImageDraw.Draw(overlay)
        _draw_boxes(draw, boxes, color, meta["cats"], show_score=show_scores)
        result  = Image.alpha_composite(base_img.copy(), overlay).convert("RGB")
        return image_to_bytes(result)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "⬇️ Ground Truth",
            data      = _single_panel_bytes(gt_anns, GT_COLOR),
            file_name = f"{dataset_name}_gt_{img_info['file_name']}",
            mime      = "image/png",
        )
    with c2:
        st.download_button(
            "⬇️ Model 1",
            data      = _single_panel_bytes(m1_filtered, M1_COLOR),
            file_name = f"{dataset_name}_model1_{img_info['file_name']}",
            mime      = "image/png",
        )
    with c3:
        st.download_button(
            "⬇️ Model 2",
            data      = _single_panel_bytes(m2_filtered, M2_COLOR),
            file_name = f"{dataset_name}_model2_{img_info['file_name']}",
            mime      = "image/png",
        )