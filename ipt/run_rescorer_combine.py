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




def rescore_dataset(args, dataset_path, 
                     run_name="", output_dir="results"):
    
    test_dir = os.path.join(dataset_path, "test")
    ann_path = os.path.join(test_dir, "_annotations.coco.json")
    # readme_path = os.path.join(dataset_path, "README.dataset.txt")
    # readme_json_path = os.path.join("./data_instr/default", f"README.dataset_{os.path.basename(dataset_path)}.json")
    readme_json_path = os.path.join(f"{args.data_instr_path}_{os.path.basename(dataset_path)}.json")
    if not os.path.isfile(ann_path):
        print(f"No test annotations found in {test_dir}, skipping.")
        return None

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
    # os.makedirs(predictions_dir, exist_ok=True)
    # os.makedirs(viz_dir, exist_ok=True)
    # os.makedirs(eval_dir, exist_ok=True)


    #Get all category ids
    ds_cat_ids = coco_gt.getCatIds()
    ds_cat_names = [coco_gt.cats[cat_id]["name"] for cat_id in ds_cat_ids]
    #Dict of cat names to ids
    cat_name2id_dict = {coco_gt.cats[cat_id]["name"]: cat_id for cat_id in ds_cat_ids}

    #Dict of cat ids to names
    cat_dict = {cat_id: coco_gt.cats[cat_id]["name"] for cat_id in ds_cat_ids}
    

    print(f"Categories in {dataset_name}: {cat_dict}")


    # Define paths for all 4 evaluation types

    eval_types = ["model", "ranking"]
    # eval_types = ["model", "ranking", "combined"]
    prediction_cache_paths = {
        eval_type: os.path.join(predictions_dir, f"predictions_{dataset_name}_{eval_type}.json") for eval_type in eval_types
    }
    

    # --- Load Predictions ---
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
                    print(f"Warning: Could not load or parse {path}. Error: {e}")
                    detections_all_by_type[eval_type] = [] # Reset if file is corrupt

                    processed_image_ids = set()  # Reset processed IDs if any file is corrupt
                    break  # No point in continuing if one file is corrupt
        
        if processed_image_ids:
            print(f"Found {len(processed_image_ids)} already processed images. GT: {len(coco_gt.dataset['images'])}.")
        else:
            raise ValueError(f"Prediction files found but no valid detections loaded for {dataset_path}. Please check the files or delete them to start fresh.")
        
    # --- End Load Predictions ---

    
    if not all(os.path.isfile(p) for p in prediction_cache_paths.values()) and len(processed_image_ids) == len(coco_gt.dataset["images"]):
        print(f"Warning! Missing predictions for {dataset_path}! GT: {len(coco_gt.dataset['images'])}, Found: {len(processed_image_ids)}. Starting evaluation with loaded predictions.")
    


    # *** Raw predictions Processing ***


    raw_predictions_path = os.path.join(output_dir, "live_results", run_name, f"{dataset_name}_live_results.json")
    with open(raw_predictions_path, "r", encoding="utf-8") as f:
        raw_predictions = json.load(f)
        

    max_score = 1.0
    min_score = 0.1

    new_ranking_detections = []
    combined_detections = []
    for prediction in raw_predictions:

        parsed_detections_ranking = prediction["parsed_detections_ranking"]
        if len(parsed_detections_ranking) == 0:
            print(f"Warning: No parsed detections found in prediction for image_id {prediction['img_id']} in {dataset_path}. Skipping this prediction.")
            continue

        class_dets = {}
        # Process each parsed detection in the ranking results
        for det in parsed_detections_ranking:
            
            if det["category_name"] in class_dets:
                class_dets[det["category_name"]].append(det)
            else:
                class_dets[det["category_name"]] = [det]


        # max_class_score = 0

        for category_name, dets in class_dets.items():

            dets = utils.apply_nms(dets, iou_threshold=0.1)

            for idx, det in enumerate(dets):

                new_rank_score = max_score - (max_score - min_score) * (idx / (len(dets) - 1)) if len(dets) > 1 else max_score 
                # new_rank_score = max_score - (max_score - min_score) * (idx / (50 - 1)) if len(dets) > 1 else max_score 

                combined_score = det["model_score"] * new_rank_score
                # combined_score = (det["model_score"] + new_rank_score) / 2.0
            

                # Append to the ranking detections list
                new_ranking_detections.append({
                    "image_id": prediction["img_id"],
                    "category_id": cat_name2id_dict.get(det["category_name"], -1),
                    "category_name": det["category_name"],
                    "bbox": det["bbox"],
                    "score": new_rank_score,
                    "model_score": det["model_score"],
                    "old_rank_score": det["rank_score"],
                    "new_rank_score": new_rank_score,
                    "combined_score": combined_score
                })

                combined_detections.append({
                    "image_id": prediction["img_id"],
                    "category_id": cat_name2id_dict.get(det["category_name"], -1),
                    "category_name": det["category_name"],
                    "bbox": det["bbox"],
                    "score": combined_score,
                    "model_score": det["model_score"],
                    "old_rank_score": det["rank_score"],
                    "new_rank_score": new_rank_score,
                    "combined_score": combined_score
                })


    detections_all_by_type["combined"] = combined_detections
    detections_all_by_type["new_ranking"] = new_ranking_detections


    images_to_process = coco_gt.dataset["images"]

    
    # # Final save after the loop completes
    # for eval_type, detections in detections_all_by_type.items():
    #     with open(prediction_cache_paths[eval_type], "w", encoding="utf-8") as f:
    #         json.dump(detections, f)
    
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



    return all_stats




# def run_single_dataset_evaluation(args):
def run_single_dataset_evaluation(args, model=None, processor=None):
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

   
    # output_dir = os.path.join(args.output_dir, args.model_name, "rf20_IPT_singleclass_rankScore", "final_instruction_eval")
    output_dir = os.path.join(args.output_dir, "final_instruction_eval")
    run_name = f"rank"

    # # Set seed for reproducibility
    utils.set_seed(args.seed)


    print("=" * 60)
    print(f"Evaluating dataset: {dataset_path}")

    ds_stats = rescore_dataset(args, dataset_path, run_name=run_name, output_dir=output_dir)



    if ds_stats is not None:
        # Print summary of results
        print("\n--- Summary of Results ---")
        # print(f"[orig_no_nms] mAP (AP50-95) for {os.path.basename(dataset_path)}: {ds_stats['orig_no_nms'][0]:.4f}")
        # print(f"[orig_with_nms] mAP (AP50-95) for {os.path.basename(dataset_path)}: {ds_stats['orig_with_nms'][0]:.4f}")
        # print(f"[vqa_no_nms] mAP (AP50-95) for {os.path.basename(dataset_path)}: {ds_stats['vqa_no_nms'][0]:.4f}")
        # print(f"[vqa_with_nms] mAP (AP50-95) for {os.path.basename(dataset_path)}: {ds_stats['vqa_with_nms'][0]:.4f}")
        print(f"[model] mAP (AP50-95) for {os.path.basename(dataset_path)}: {ds_stats['model'][0]:.4f}")
        print(f"[ranking] mAP (AP50-95) for {os.path.basename(dataset_path)}: {ds_stats['ranking'][0]:.4f}")

        #AR@1
        # print(f"[orig_no_nms] AR@1 for {os.path.basename(dataset_path)}: {ds_stats['orig_no_nms'][6]:.4f}")
        # print(f"[orig_with_nms] AR@1 for {os.path.basename(dataset_path)}: {ds_stats['orig_with_nms'][6]:.4f}")
        # print(f"[vqa_no_nms] AR@1 for {os.path.basename(dataset_path)}: {ds_stats['vqa_no_nms'][6]:.4f}")
        # print(f"[vqa_with_nms] AR@1 for {os.path.basename(dataset_path)}: {ds_stats['vqa_with_nms'][6]:.4f}")
        print(f"[model] AR@1 for {os.path.basename(dataset_path)}: {ds_stats['model'][6]:.4f}")
        print(f"[ranking] AR@1 for {os.path.basename(dataset_path)}: {ds_stats['ranking'][6]:.4f}")

    else:
        print(f"Evaluation failed for {dataset_path}")



# cmd: 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen3-VL-235B-A22B-Instruct", help='model name e.g., Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-72B-Instruct, Qwen3-VL-8B-Instruct, Qwen3-VL-30B-A3B-Instruct, Qwen3-VL-235B-A22B-Instruct]')
    # parser.add_argument("--no_instructions", action="store_true", help="Run inference with no instructions")
    # parser.add_argument("--few_shot", action="store_true", help="Use 3 random few-shot examples from test set")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to a single dataset to evaluate. If not set, all datasets will be evaluated in parallel.")
    parser.add_argument("--output_dir", type=str, default="results/rf100vl-zeroshot/rf20_IPT_singleclass_vqaScore_withNMS", help="Directory to save results and visuals.")
    parser.add_argument("--data_instr_path", type=str, default="./data_instr/default/README.dataset", help="Directory to save results and visuals.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None, help='List of GPU IDs to use for processing. e.g. --gpu_ids 0 1 4')
    parser.add_argument('--vqa_batch_size', type=int, default=8, help='Batch size for VQA scoring of candidate masks.')
    parser.add_argument("--vqa_rescore", action="store_true", help="Use VQA-based re-scoring of candidate masks")
    parser.add_argument("--rank_rescore", action="store_true", help="Use Rank-based class re-scoring of candidate masks")
    parser.add_argument("--rating_rescore", action="store_true", help="Use Rating-based class re-scoring of candidate masks")
    parser.add_argument("--siglip_rescore", action="store_true", help="Use SigLip-based re-scoring of candidate masks")
    # parser.add_argument("--apply_nms", action="store_true", help="Apply Non-Maximum Suppression to detections.")
    # parser.add_argument("--nms_threshold", type=float, default=0.5, help="IoU threshold for Non-Maximum Suppression.")
    # parser.add_argument("--class_rescore", action="store_true", help="Use VQA-based class re-scoring of candidate masks")
    parser.add_argument("--device_map_auto", action="store_true", help="Use device_map='auto' for model loading. Overrides --qwen_device if set.")

    args = parser.parse_args()

    run_single_dataset_evaluation(args)
    