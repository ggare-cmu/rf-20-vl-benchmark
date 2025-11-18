'''
Run cmd: CUDA_VISIBLE_DEVICES=0,1 python ipt/run_bench_singleclass_evaluator.py --model_name Qwen2.5-VL-7B-Instruct --vqa_rescore --apply_nms --nms_threshold 0.5 --data_instr_path results/rf100vl_IPT/Qwen2.5-VL-7B-Instruct/rf20_IPT_singleclass_vqaScore_withNMS/iterative_prompt_refinement/all_refined_class_instructions --output_dir results/rf100vl_IPT_eval_tmp/rf20_IPT_singleclass_vqaScore_withNMS_tmp --vqa_batch_size 1 --dataset_path wb-prova
'''

import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import json
import torch

from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


import time
import gc

import numpy as np
import argparse


import ipt_utils as utils




def evaluate_dataset(args, model, processor, dataset_path, no_instructions, few_shot_examples=False, 
                     run_name="", output_dir="results", max_samples=None, sigclip_pipe=None):
    test_dir = os.path.join(dataset_path, "test")
    ann_path = os.path.join(test_dir, "_annotations.coco.json")
    # readme_path = os.path.join(dataset_path, "README.dataset.txt")
    # readme_json_path = os.path.join("./data_instr/default", f"README.dataset_{os.path.basename(dataset_path)}.json")
    readme_json_path = os.path.join(f"{args.data_instr_path}_{os.path.basename(dataset_path)}.json")
    if not os.path.isfile(ann_path):
        print(f"No test annotations found in {test_dir}, skipping.")
        return None
    if few_shot_examples:
        # few_shot_dict = utils.build_few_shot_dict(dataset_path, examples_per_class=2)
        few_shot_dict = utils.build_few_shot_dict(dataset_path, examples_per_class=10)

        assert sum([len(v) for k,v in few_shot_dict.items()]) == len([f for f in os.listdir(test_dir.replace('test', 'train')) if f.split('.')[-1] in ['.png', 'jpeg', 'jpg']]), "Few-shot examples count does not match number of training images!"
    
        few_shot_samples = []
        [few_shot_samples.extend(v) for k,v in few_shot_dict.items() if len(v) > 0]
        print(f"Built few-shot examples dictionary with {len(few_shot_dict)} categories and {len(few_shot_samples)} total examples for {dataset_path}.")
    else:
        few_shot_dict = None

    dataset_instructions_json = {}
    if os.path.isfile(readme_json_path):
        with open(readme_json_path, "r", encoding="utf-8") as f:
            dataset_instructions_json = json.load(f)
    print(f"\n\n\nLoaded dataset instructions for {dataset_path} from {readme_json_path}: \n{dataset_instructions_json}\n\n\n\n")

    coco_gt = COCO(ann_path)

    dataset_name = os.path.basename(dataset_path)
    predictions_dir = os.path.join(output_dir, "predictions", run_name)
    viz_dir = os.path.join(output_dir, "visuals", run_name)
    eval_dir = os.path.join(output_dir, "evaluations", run_name)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Define paths for all 4 evaluation types
    eval_types = ["orig_no_nms", "orig_with_nms", "vqa_no_nms", "vqa_with_nms"]
    prediction_cache_paths = {
        eval_type: os.path.join(predictions_dir, f"predictions_{dataset_name}_{eval_type}.json") for eval_type in eval_types
    }
    
    # --- Resume Logic ---
    detections_all_by_type = {eval_type: [] for eval_type in eval_types}
    processed_image_ids = set()

    # Check if any prediction file exists to attempt resuming
    if any(os.path.isfile(p) for p in prediction_cache_paths.values()):
        print(f"Attempting to resume from cached predictions for {dataset_path}")
    
        for eval_type, path in prediction_cache_paths.items():
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        loaded_detections = json.load(f)
                        detections_all_by_type[eval_type] = loaded_detections
                        # Add the image_ids from the loaded detections to our set
                        for det in loaded_detections:
                            processed_image_ids.add(det['image_id'])
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not load or parse {path}. Starting this eval type from scratch. Error: {e}")
                    detections_all_by_type[eval_type] = [] # Reset if file is corrupt

                    processed_image_ids = set()  # Reset processed IDs if any file is corrupt
                    break  # No point in continuing if one file is corrupt
        
        if processed_image_ids:
            print(f"Resuming. Found {len(processed_image_ids)} already processed images.")

    # --- End Resume Logic ---

    # Only run inference if not all predictions are loaded and complete
    # if not all(os.path.isfile(p) for p in prediction_cache_paths.values()) or not processed_image_ids:

    if all(os.path.isfile(p) for p in prediction_cache_paths.values()) and len(processed_image_ids) == len(coco_gt.dataset["images"]):
        print(f"All predictions already exist for {dataset_path}, skipping inference and using cached predictions.")
    else:

        
        total_count = 0

        images_to_process = coco_gt.dataset["images"]
        if max_samples is not None:
            # Ensure we don't request more samples than available
            num_to_process = min(max_samples, len(images_to_process))
            images_to_process = images_to_process[:num_to_process]

        #Get all category ids
        ds_cat_ids = coco_gt.getCatIds()
        ds_cat_names = [coco_gt.cats[cat_id]["name"] for cat_id in ds_cat_ids]
        #Dict of cat names to ids
        cat_name2id_dict = {coco_gt.cats[cat_id]["name"]: cat_id for cat_id in ds_cat_ids}

        #Dict of cat ids to names
        cat_dict = {cat_id: coco_gt.cats[cat_id]["name"] for cat_id in ds_cat_ids}
        

        print(f"Categories in {dataset_name}: {cat_dict}")

        with torch.inference_mode():
            for img_info in tqdm(images_to_process, desc=f"Processing images in {os.path.basename(dataset_path)}"):
                # Skip images that have already been processed
                if img_info['id'] in processed_image_ids:
                    print(f"Skipping already processed image_id: {img_info['id']}")
                    total_count += 1
                    continue

                img_id = img_info["id"]
                img_filename = img_info["file_name"]
                image_path = os.path.join(test_dir, img_filename)
    
                if not os.path.isfile(image_path):
                    print(f"Image file not found: {image_path}. Skipping.")
                    continue
    
                ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
                anns = coco_gt.loadAnns(ann_ids)
                
                cat_ids_for_image = set(ann["category_id"] for ann in anns)
                print(f"Image {img_filename} has categories: {[coco_gt.cats[cat_id]['name'] for cat_id in cat_ids_for_image]}")
                
                
                raw_output, few_shot_examples_used, all_detections = utils.run_inference_on_single_image( #grg_changed
                    args,
                    model, processor,
                    image_path=image_path,
                    dataset_instructions_json = dataset_instructions_json,
                    class_name_list=ds_cat_names, #GRG: Pass the entire list of category names
                    no_instructions=no_instructions,
                    few_shot_dict=few_shot_dict,
                    output_dir=output_dir,
                    sigclip_pipe=sigclip_pipe,
                )

                for eval_type, detections in all_detections.items():
                    for det in detections:
                        detections_all_by_type[eval_type].append({
                            "image_id": img_id,
                            "category_id": cat_name2id_dict.get(det["category_name"], -1),
                            "bbox": det["bbox"],
                            "score": det["score"]
                        })
                
                total_count += 1

                # --- Incremental Save ---
                if total_count % 5 == 0:
                    print(f"\nSaving intermediate results at image {total_count + 1}/{len(images_to_process)}...")
                    for eval_type, detections in detections_all_by_type.items():
                        with open(prediction_cache_paths[eval_type], "w", encoding="utf-8") as f:
                            json.dump(detections, f)
                
                
                # Yield results for live updates
                yield {
                    "img_id": img_id,
                    "image_path": image_path,
                    "gt_bboxes": [ann["bbox"] for ann in anns],
                    # "pred_bboxes": [det["bbox"] for det in all_detections["vqa_with_nms"]],
                    "pred_bboxes": [det["bbox"] for det in all_detections["vqa_no_nms"]],
                    "raw_output": raw_output,
                    "parsed_detections_org_before_nms": all_detections["orig_no_nms"],
                    "parsed_detections_org_with_nms": all_detections["orig_with_nms"],
                    "parsed_detections_before_nms": all_detections["vqa_no_nms"],
                    "parsed_detections": all_detections["vqa_with_nms"],
                    "gt_anns": anns,
                    "cat_dict": cat_dict,
                    "few_shot_examples_used": few_shot_examples_used,
                }

        del raw_output
        torch.cuda.empty_cache()
        gc.collect()

        # Final save after the loop completes
        for eval_type, detections in detections_all_by_type.items():
            with open(prediction_cache_paths[eval_type], "w", encoding="utf-8") as f:
                json.dump(detections, f)
    
    # --- Run evaluation for all 4 types ---
    all_stats = {}
    for eval_type, detections in detections_all_by_type.items():
        print(f"\n--- Evaluating: {eval_type} ---")
        if not detections:
            print(f"No detections for {eval_type} in {dataset_path}.")
            all_stats[eval_type] = [0.0] * 12
            continue

        coco_dt = coco_gt.loadRes(detections)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        all_stats[eval_type] = coco_eval.stats


    eval_results_path = os.path.join(
        eval_dir, f"evaluation_{dataset_name}.json"
    )
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_stats = {
        # eval_type: stats.tolist() for eval_type, stats in all_stats.items()
        eval_type: stats.tolist() if hasattr(stats, "tolist") else stats for eval_type, stats in all_stats.items()
    }

    with open(eval_results_path, "w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, indent=2)

    print(f"Saved evaluation results to {eval_results_path}")

    return all_stats




def run_single_dataset_evaluation(args):
    """
    Runs evaluation for a single dataset. This function is called by the dispatcher.
    """

    root_dir = "./datasets/rf100-vl-fsod/"
    if not os.path.isdir(root_dir):
        print(f"Root directory not found: {root_dir}")
        return
    
    dataset_path = os.path.join(root_dir, args.dataset_path)

    if not dataset_path or not os.path.isdir(dataset_path):
        print(f"Error: Invalid or missing --dataset_path: {dataset_path}")
        return

    run_modes = []
    if args.no_instructions:
        run_modes.append("noinstr")
    if args.few_shot:
        run_modes.append("fewshot")
    if args.vqa_rescore:
        run_modes.append("vqa")
    if args.class_rescore:
        run_modes.append("cls_rescore")
    # Add NMS threshold to run name to differentiate runs
    run_modes.append(f"nms{args.nms_threshold}")
    run_name = "_".join(run_modes) if run_modes else "default"

    # Set seed for reproducibility
    utils.set_seed(args.seed)

    # os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using model: {args.model_name}")

    model, processor = utils.load_qwen_model(args.model_name)

    if args.siglip_rescore:
        sigclip_pipe = utils.load_sigclip_pipeline()
        print("Loaded SigClip pipeline for confidence scoring.")


    print("=" * 60)
    print(f"Evaluating dataset: {dataset_path}")

    eval_generator = evaluate_dataset(args, model, processor, dataset_path, no_instructions=args.no_instructions, few_shot_examples=args.few_shot, 
                                      run_name=run_name, output_dir=args.output_dir,
                                      sigclip_pipe=sigclip_pipe if args.siglip_rescore else None)


    # Collect live results yielded by the generator and save them to disk periodically
    live_results = []
    ds_stats = None

    def _save_live_results_snapshot(live_results, suffix=""):
        """
        Append the provided `live_results` batch to a master JSONL and optionally
        write a per-part JSONL (when suffix is provided). If suffix is empty,
        this is treated as the final save: append remaining items and write a
        pretty JSON by reading the master JSONL.
        """
        try:
            dataset_basename = os.path.basename(dataset_path.rstrip('/'))
            save_dir = os.path.join(args.output_dir, "live_results", run_name)
            os.makedirs(save_dir, exist_ok=True)

            master_jsonl = os.path.join(save_dir, f"{dataset_basename}_live_results.jsonl")
            part_jsonl = os.path.join(save_dir, f"{dataset_basename}_live_results{suffix}.jsonl") if suffix else None
            pretty_json = os.path.join(save_dir, f"{dataset_basename}_live_results{suffix}.json") if suffix else os.path.join(save_dir, f"{dataset_basename}_live_results.json")

            # Append batch to master jsonl (create if missing)
            with open(master_jsonl, "a", encoding="utf-8") as fjsonl:
                for rec in live_results:
                    json.dump(rec, fjsonl)
                    fjsonl.write("\n")

            # Also write a part file for this batch if requested (useful for quick inspection)
            if part_jsonl is not None:
                with open(part_jsonl, "w", encoding="utf-8") as fpart:
                    for rec in live_results:
                        json.dump(rec, fpart)
                        fpart.write("\n")

            # If this is the final snapshot (no suffix), build a pretty JSON by reading master jsonl
            if not suffix:
                all_recs = []
                try:
                    with open(master_jsonl, "r", encoding="utf-8") as fmaster:
                        for line in fmaster:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                all_recs.append(json.loads(line))
                            except Exception:
                                # skip malformed lines
                                continue
                except FileNotFoundError:
                    all_recs = []

                with open(pretty_json, "w", encoding="utf-8") as fjson:
                    json.dump(all_recs, fjson, indent=2)

                print(f"Saved final {len(all_recs)} live results to {save_dir}")
            else:
                print(f"Appended {len(live_results)} live results to {master_jsonl} and saved part {part_jsonl}")

        except Exception as e:
            print(f"Failed to save live results: {e}")

    try:
        counter = 0
        while True:
            try:
                item = next(eval_generator)
                # Append the yielded result to the current buffer
                live_results.append(item)
                counter += 1

                # Periodically save every 20 iterations: append the buffered records to
                # the master JSONL, write a per-part file for quick inspection, then
                # clear the in-memory buffer to free memory.
                if counter % 20 == 0:
                    _save_live_results_snapshot(live_results, suffix=f"_part{counter}")
                    # Clear the in-memory buffer after persisting
                    live_results.clear()

            except StopIteration as e:
                # Generator finished; capture return value (final stats)
                ds_stats = e.value
                break
    except Exception as e:
        print(f"Error while consuming eval_generator: {e}")

    # Final save: append any remaining buffered records and produce a pretty JSON
    if live_results:
        _save_live_results_snapshot(live_results, suffix="")
    else:
        # Even if buffer is empty, ensure final pretty JSON exists by calling with empty list
        _save_live_results_snapshot([], suffix="")

    if ds_stats is not None:
        # Print summary of results
        print("\n--- Summary of Results ---")
        print(f"[orig_no_nms] mAP (AP50-95) for {os.path.basename(dataset_path)}: {ds_stats['orig_no_nms'][0]:.4f}")
        print(f"[orig_with_nms] mAP (AP50-95) for {os.path.basename(dataset_path)}: {ds_stats['orig_with_nms'][0]:.4f}")
        print(f"[vqa_no_nms] mAP (AP50-95) for {os.path.basename(dataset_path)}: {ds_stats['vqa_no_nms'][0]:.4f}")
        print(f"[vqa_with_nms] mAP (AP50-95) for {os.path.basename(dataset_path)}: {ds_stats['vqa_with_nms'][0]:.4f}")

        #AR@1
        print(f"[orig_no_nms] AR@1 for {os.path.basename(dataset_path)}: {ds_stats['orig_no_nms'][6]:.4f}")
        print(f"[orig_with_nms] AR@1 for {os.path.basename(dataset_path)}: {ds_stats['orig_with_nms'][6]:.4f}")
        print(f"[vqa_no_nms] AR@1 for {os.path.basename(dataset_path)}: {ds_stats['vqa_no_nms'][6]:.4f}")
        print(f"[vqa_with_nms] AR@1 for {os.path.basename(dataset_path)}: {ds_stats['vqa_with_nms'][6]:.4f}")

    else:
        print(f"Evaluation failed for {dataset_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen3-VL-235B-A22B-Instruct", help='model name e.g., Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-72B-Instruct, Qwen3-VL-8B-Instruct, Qwen3-VL-30B-A3B-Instruct, Qwen3-VL-235B-A22B-Instruct]')
    parser.add_argument("--no_instructions", action="store_true", help="Run inference with no instructions")
    parser.add_argument("--few_shot", action="store_true", help="Use 3 random few-shot examples from test set")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to a single dataset to evaluate. If not set, all datasets will be evaluated in parallel.")
    parser.add_argument("--output_dir", type=str, default="results/rf100vl-zeroshot/rf20_IPT_singleclass_vqaScore_withNMS", help="Directory to save results and visuals.")
    parser.add_argument("--data_instr_path", type=str, default="./data_instr/default/README.dataset", help="Directory to save results and visuals.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None, help='List of GPU IDs to use for processing. e.g. --gpu_ids 0 1 4')
    parser.add_argument('--vqa_batch_size', type=int, default=8, help='Batch size for VQA scoring of candidate masks.')
    parser.add_argument("--vqa_rescore", action="store_true", help="Use VQA-based re-scoring of candidate masks")
    parser.add_argument("--siglip_rescore", action="store_true", help="Use SigLip-based re-scoring of candidate masks")
    parser.add_argument("--apply_nms", action="store_true", help="Apply Non-Maximum Suppression to detections.")
    parser.add_argument("--nms_threshold", type=float, default=0.5, help="IoU threshold for Non-Maximum Suppression.")
    parser.add_argument("--class_rescore", action="store_true", help="Use VQA-based class re-scoring of candidate masks")
    parser.add_argument("--device_map_auto", action="store_true", help="Use device_map='auto' for model loading. Overrides --qwen_device if set.")

    args = parser.parse_args()

    run_single_dataset_evaluation(args)
    