"""
Streamlit app for visualizing prediction JSON(s) produced by
`run_bench_singleclass_VQAscoring_webUI_multimetrics.py`.

Features:
- Upload a predictions JSON file (or point to a local path).
- Optionally provide a COCO annotation JSON to map image IDs and show ground-truth boxes.
- Optionally provide a local images directory to resolve image files.
- Configure top-K boxes, show scores, and output directory.
- Generate visualizations and display thumbnails in the Streamlit UI.

Run:
streamlit run code/visualize_predictions_streamlit.py

"""

import glob
import streamlit as st
import json
import os
import tempfile
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

try:
    from pycocotools.coco import COCO
except Exception:
    COCO = None

st.set_page_config(layout="wide", page_title="Prediction Visualizer")

# --- Helper functions (adapted from CLI visualizer) ---

def load_json_from_path_or_uploaded(obj):
    """Accept either a file path (str) or an uploaded file-like object from Streamlit."""
    if obj is None:
        return None
    if isinstance(obj, str):
        with open(obj, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # uploaded file-like object
        try:
            return json.load(obj)
        except Exception:
            obj.seek(0)
            text = obj.read().decode("utf-8") if isinstance(obj.read(), bytes) else obj.read()
            return json.loads(text)


def normalize_bbox(item):
    if not item:
        return None
    if "bbox" in item:
        b = item["bbox"]
        if len(b) == 4:
            return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    if "bbox_2d" in item:
        b = item["bbox_2d"]
        if len(b) == 4:
            # If bbox_2d given as [x1,y1,x2,y2], convert to xywh
            x0, y0, x1, y1 = [float(x) for x in b]
            if x1 > x0 and y1 > y0:
                return [x0, y0, x1 - x0, y1 - y0]
            return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    if "bbox_xyxy" in item:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        return [float(x1), float(y1), float(x2) - float(x1), float(y2) - float(y1)]
    if "bbox_xywh" in item:
        b = item["bbox_xywh"]
        return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    return None


def group_predictions_by_image(predictions):
    grouped = defaultdict(list)
    for p in predictions:
        image_id = None
        for k in ("image_id", "img_id", "file_name", "image_path", "image"):
            if k in p:
                image_id = p[k]
                break
        if isinstance(image_id, dict) and "file_name" in image_id:
            image_id = image_id["file_name"]
        grouped[image_id].append(p)
    return grouped


def draw_detections_on_image(pil_image, detections, show_scores=True, coco=None, max_boxes=None, font=None, gt=False):
    draw = ImageDraw.Draw(pil_image)
    w, h = pil_image.size
    if font is None:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", max(12, int(h * 0.03)))
        except Exception:
            font = ImageFont.load_default()

    def _get_text_size(draw_obj, text, font_obj):
        """Return (width, height) for rendered text using available APIs."""
        try:
            bbox = draw_obj.textbbox((0, 0), text, font=font_obj)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])
        except Exception:
            pass
        try:
            return font_obj.getsize(text)
        except Exception:
            pass
        try:
            return draw_obj.textsize(text, font=font_obj)
        except Exception:
            return (len(text) * 6, 10)

    def score_of(d):
        # return float(d.get("score", d.get("conf", d.get("confidence", 0.0))))
        return d["score"]

    if not gt:
        detections_sorted = sorted(detections, key=score_of, reverse=True)
    else:
        detections_sorted = detections
        
    if max_boxes:
        detections_sorted = detections_sorted[:max_boxes]

    for det in detections_sorted:
        bbox = normalize_bbox(det)
        if bbox is None:
            if "bbox" in det and isinstance(det["bbox"], (list, tuple)) and len(det["bbox"]) == 4:
                bbox = [float(x) for x in det["bbox"]]
            else:
                continue
        x, y, bw, bh = bbox
        x1, y1, x2, y2 = x, y, x + bw, y + bh
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # cls_name = coco.cats[det["category_id"]]["name"] if coco else str(det["category_id"])
        if gt:
            cls_name = det["category_name"]
        else:
            cls_name = coco.cats[det["category_id"]]["name"] if coco else str(det["category_id"])
        
        # deterministic color
        col_hash = abs(hash(cls_name)) if cls_name else (1 << 24)
        color = ((col_hash >> 16) & 255, (col_hash >> 8) & 255, col_hash & 255)
        # GT boxes green override
        if gt:
            color = (0, 255, 0)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, int(min(w, h) * 0.005)))
        if show_scores and not gt:
            score = score_of(det)
            txt = f"{cls_name} ({score:.2f})" if cls_name else f"{score:.2f}"
        else:
            txt = f"{cls_name}" if cls_name else ""
        text_size = _get_text_size(draw, txt, font)
        text_bg = [x1, max(0, y1 - text_size[1] - 2), x1 + text_size[0] + 4, y1]
        draw.rectangle(text_bg, fill=color)
        draw.text((x1 + 2, max(0, y1 - text_size[1] - 1)), txt, fill=(255, 255, 255), font=font)

    return pil_image


def map_image_id_to_file(coco, image_id):
    if coco is None:
        return None
    try:
        if isinstance(image_id, int) or (isinstance(image_id, str) and image_id.isdigit()):
            img_info = coco.loadImgs([int(image_id)])[0]
            return img_info["file_name"], int(image_id)
        for img in coco.dataset.get("images", []):
            if img.get("file_name") == image_id or os.path.basename(img.get("file_name", "")) == image_id:
                return img.get("file_name"), img.get("id")
        return None
    except Exception:
        return None

# --- Streamlit UI ---

st.title("Prediction Visualizer — Streamlit")

with st.sidebar:
    st.header("Inputs")

    # results_dir = st.text_input("Results directory", value="./results/rf100vl_zeroshot/rf20_singleclass_codePrompt_vqaScore_nms0.5/")
    # results_dir = st.text_input("Results directory", value="./results/rf100vl_IPT_Gemini/gemini-2.5-pro-preview-03-25/rf20_IPT_singleclass_Novqa/iterative_prompt_refinement/aerial-airport/predictions/class_airplane_ipt_iter_4/")
    results_dir = st.text_input("Results directory", value="./results/lambda_results/132.145.195.234/results/rf100vl_IPT/Qwen3-VL-30B-A3B-Instruct/rf20_IPT_singleclass_vqaScore_withNMS/final_instruction_eval/")

    # Prediction JSON files
    pred_json_pattern = os.path.join(results_dir, 'predictions', '*', 'predictions*.json') #default - implies with no_instructions
    # pred_json_pattern = os.path.join(results_dir, 'predictions*.json') #default - implies with no_instructions
    pred_jsons_files = glob.glob(pred_json_pattern)

    pred_jsons_files = sorted(pred_jsons_files)

    pred_json_dict = {os.path.basename(f).replace(".json", "").replace("predictions_", ""): f for f in pred_jsons_files} 

    #Selectbox to choose among prediction json files
    # selected_pred_json = st.selectbox("Select prediction JSON file", options=pred_jsons_files)
    selected_pred_json = st.selectbox("Select prediction JSON file", options=list(pred_json_dict.keys()))
    preds_path = pred_json_dict[selected_pred_json]

    # preds_path = st.text_input("Or local predictions file/folder path (leave blank to use uploads)", value="/home/grg/Research/VLMattributeClassifier/results/rf100vl_zeroshot/rf20_singleclass_codePrompt_vqaScore_nms0.5/predictions/vqa_nms0.5/predictions_actions_vqa_no_nms.json")

    uploaded_preds = st.file_uploader("Upload prediction JSON file(s)", type=["json"], accept_multiple_files=True)
    
    dataset_name = preds_path.split("/")[-1].split("_")[1]
    st.session_state.dataset_name = dataset_name

    coco_path = os.path.join("./datasets/rf100-vl-fsod/", dataset_name, "test", "_annotations.coco.json")
    # coco_path = os.path.join("./datasets/rf100-vl-fsod/", dataset_name, "train", "_annotations.coco.json")
    if os.path.isfile(coco_path):
        st.session_state.coco_path = coco_path

    images_dir = os.path.join("./datasets/rf100-vl-fsod/", dataset_name, "test")
    # images_dir = os.path.join("./datasets/rf100-vl-fsod/", dataset_name, "train")
    if os.path.isdir(images_dir):
        st.session_state.images_dir = images_dir

    # uploaded_coco = st.file_uploader("Upload COCO annotation JSON (optional)", type=["json"])
    # coco_path = st.text_input("Or local COCO annotation file path", value="/home/grg/Research/VLMattributeClassifier/datasets/rf100-vl-fsod/actions/test/_annotations.coco.json")

    # images_dir = st.text_input("Local images directory (optional)", value="/home/grg/Research/VLMattributeClassifier/datasets/rf100-vl-fsod/actions/test/")
    


    topk = st.number_input("Top-K predictions per image", value=10, min_value=1, max_value=500, step=1)
    show_scores = st.checkbox("Show scores on boxes", value=True)

    output_dir = os.path.join(results_dir, 'visuals_pred_bbox_GT', f"{selected_pred_json}_top{topk}")
    st.session_state.output_dir = output_dir

    # output_dir = st.text_input("Output directory (server-side)", value="results/visuals_streamlit")
    

    run_button = st.button("Generate visuals")
   
st.markdown("---")

# ensure output exists
if run_button:
    os.makedirs(output_dir, exist_ok=True)

    coco = None
    coco_tempfile = None
    # if uploaded_coco:
    #     # save to temporary file for pycocotools to load
    #     try:
    #         cof = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    #         cof.write(uploaded_coco.getvalue())
    #         cof.flush()
    #         coco_tempfile = cof.name
    #         cof.close()
    #         if COCO is None:
    #             st.warning("pycocotools not available — ground-truth display disabled. Install pycocotools to enable COCO support.")
    #         else:
    #             coco = COCO(coco_tempfile)
    #     except Exception as e:
    #         st.error(f"Failed to load uploaded COCO json: {e}")
    # elif coco_path:
    if coco_path:
        if os.path.isfile(coco_path):
            if COCO is None:
                st.warning("pycocotools not available — ground-truth display disabled. Install pycocotools to enable COCO support.")
            else:
                coco = COCO(coco_path)
        else:
            st.warning("Provided COCO path does not exist.")

    # Collect prediction file sources
    pred_sources = []
    if uploaded_preds:
        # save each uploaded file to a temp file and add path
        for up in uploaded_preds:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            tmp.write(up.getvalue())
            tmp.flush()
            tmp.close()
            pred_sources.append(tmp.name)
    if preds_path:
        if os.path.isdir(preds_path):
            for fn in os.listdir(preds_path):
                if fn.lower().endswith('.json'):
                    pred_sources.append(os.path.join(preds_path, fn))
        elif os.path.isfile(preds_path):
            pred_sources.append(preds_path)
        else:
            st.warning("preds path does not exist")

    if not pred_sources:
        st.warning("No predictions provided. Upload files or give a path.")
    else:
        progress_bar = st.progress(0)
        total_files = len(pred_sources)
        all_saved = []
        for idx, pf in enumerate(pred_sources, start=1):
            st.write(f"Processing {os.path.basename(pf)} ({idx}/{total_files})")
            try:
                preds = load_json_from_path_or_uploaded(pf)
            except Exception as e:
                st.error(f"Failed to load {pf}: {e}")
                continue

            if isinstance(preds, dict) and "annotations" in preds and isinstance(preds["annotations"], list):
                pred_list = preds["annotations"]
            elif isinstance(preds, list):
                pred_list = preds
            else:
                pred_list = []
                if isinstance(preds, dict):
                    for v in preds.values():
                        if isinstance(v, list):
                            pred_list.extend(v)

            grouped = group_predictions_by_image(pred_list)
            st.write(f"Found predictions for {len(grouped)} images")

            # display/save images
            
            # Live preview placeholders for this prediction file
            st.markdown(f"**Live preview for {os.path.basename(pf)}**")
            preview_col, info_col = st.columns([2, 1])
            preview_placeholder = preview_col.empty()
            info_placeholder = info_col.empty()
            
            
            saved_for_file = []
            for image_id, dets in grouped.items():
                image_path = None
                coco_img_id = None
                if coco is not None and image_id is not None:
                    mapping = map_image_id_to_file(coco, image_id)
                    if mapping:
                        file_name, coco_img_id = mapping
                        if images_dir:
                            candidate = os.path.join(images_dir, file_name)
                            if os.path.isfile(candidate):
                                image_path = candidate
                            else:
                                if os.path.isfile(file_name):
                                    image_path = file_name
                                else:
                                    for root, _, files in os.walk(images_dir):
                                        if file_name in files:
                                            image_path = os.path.join(root, file_name)
                                            break
                        else:
                            if os.path.isfile(file_name):
                                image_path = file_name

                if image_path is None and isinstance(image_id, str):
                    candidate = image_id
                    if images_dir and not os.path.isabs(candidate):
                        candidate = os.path.join(images_dir, candidate)
                    if os.path.isfile(candidate):
                        image_path = candidate

                if image_path is None:
                    for d in dets:
                        for key in ("image_path", "file_name", "filename"):
                            if key in d:
                                candidate = d[key]
                                if images_dir and not os.path.isabs(candidate):
                                    candidate = os.path.join(images_dir, candidate)
                                if os.path.isfile(candidate):
                                    image_path = candidate
                                    break
                        if image_path:
                            break

                if image_path is None:
                    if images_dir and isinstance(image_id, str):
                        for root, _, files in os.walk(images_dir):
                            for fn in files:
                                if fn == image_id or fn == os.path.basename(image_id):
                                    image_path = os.path.join(root, fn)
                                    break
                            if image_path:
                                break

                if image_path is None:
                    # skip if cannot resolve image
                    continue

                try:
                    pil = Image.open(image_path).convert("RGB")
                except Exception:
                    continue

                # # draw preds
                # pil = draw_detections_on_image(pil, dets, show_scores=show_scores, coco=coco, max_boxes=topk)

                # draw GT if available
                if coco is not None and coco_img_id is not None:
                    ann_ids = coco.getAnnIds(imgIds=[coco_img_id])
                    anns = coco.loadAnns(ann_ids) if ann_ids else []
                    gt_dets = []
                    for a in anns:
                        if "bbox" in a:
                            gt_dets.append({"bbox": a["bbox"], "category_name": coco.cats[a["category_id"]]["name"]})
                    if gt_dets:
                        pil = draw_detections_on_image(pil, gt_dets, show_scores=False, coco=coco, max_boxes=None, gt=True)

                # # draw preds
                # pil = draw_detections_on_image(pil, dets, show_scores=show_scores, coco=coco, max_boxes=topk)


                # Live update: show the image and some metadata as soon as it's rendered
                try:
                    preview_placeholder.image(pil, caption=f"{os.path.basename(image_path)} ({image_id})", width="stretch")
                except Exception:
                    # fallback if image too large or streaming fails
                    preview_placeholder.image(pil.resize((800, int(800 * pil.height / pil.width))), caption=f"{os.path.basename(image_path)} ({image_id})")

                # Show metadata: top-k detections summary
                try:
                    topk_list = []
                    for d in (dets[:min(len(dets), topk)]):
                        topk_list.append({
                            "bbox": d.get("bbox", d.get("bbox_2d")),
                            "score": d.get("score", d.get("conf", d.get("confidence"))),
                            "class_name": coco.cats[d["category_id"]]["name"]
                        })
                    sorted_topk = sorted(topk_list, key=lambda x: x["score"], reverse=True)
                    info_placeholder.json({
                        "image_id": image_id,
                        "image_path": image_path,
                        "num_detections": len(dets),
                        "top_detections": sorted_topk,
                    })
                except Exception:
                    info_placeholder.write(f"Image: {image_id} — {len(dets)} detections")

                
                # save
                base_name = str(image_id) if image_id is not None else os.path.splitext(os.path.basename(image_path))[0]
                out_name = f"{base_name}_{os.path.basename(pf).replace('.','_')}.png"
                out_path = os.path.join(output_dir, out_name)
                pil.save(out_path)
                saved_for_file.append(out_path)

            # show thumbnails for this predictions file
            if saved_for_file:
                st.write(f"Saved {len(saved_for_file)} images for {os.path.basename(pf)}")
                cols = st.columns(3)
                for i, v in enumerate(saved_for_file):
                    with cols[i % 3]:
                        st.image(v, caption=os.path.basename(v), width="stretch")
                all_saved.extend(saved_for_file)

            progress_bar.progress(int(100.0 * idx / total_files))

        if all_saved:
            st.success(f"Saved {len(all_saved)} visuals to {output_dir}")
            st.info("Open the output directory on the server to view all files or download single images from the UI thumbnails.")

    # cleanup temporary coco file if created
    if 'coco_tempfile' in locals() and coco_tempfile:
        try:
            os.unlink(coco_tempfile)
        except Exception:
            pass

st.markdown("---")
st.write("Hint: if you run Streamlit on the same machine that produced the predictions, prefer using local paths (images directory + predictions folder) for faster operation.")


