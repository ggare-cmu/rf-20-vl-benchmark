'''
Run cmd: CUDA_VISIBLE_DEVICES=0,1 python ipt/run_bench_singleclass_IPT.py --model_name Qwen2.5-VL-7B-Instruct --ipt_mode --vqa_rescore --apply_nms --nms_threshold 0.5 --num_ipt_iterations 10 --output_dir results/rf100vl_IPT_tmp/rf20_IPT_singleclass_vqaScore_withNMS --vqa_batch_size 1 --dataset_path wb-prova
'''

import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import json
import torch
from PIL import Image
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import time
import gc

import numpy as np
import argparse
import re
import random

import run_bench_singleclass_evaluator as evaluator

import ipt_utils as utils




def get_seed_state():
    """Returns the current random seed state."""
    return {
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'torch_cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }

def set_seed_from_state(seed_state):
    """Sets the random seed state from a given state."""
    random.setstate(seed_state['python_random_state'])
    np.random.set_state(seed_state['numpy_random_state'])
    torch.set_rng_state(seed_state['torch_random_state'])
    if torch.cuda.is_available() and seed_state['torch_cuda_random_state'] is not None:
        torch.cuda.set_rng_state_all(seed_state['torch_cuda_random_state'])




def evaluate_dataset(args, model, processor, dataset_path, no_instructions, few_shot_examples=False, run_name="", output_dir="results", 
                     eval_class_name=None, eval_cat_id=None, 
                     max_samples=None, 
                     dataset_instructions_override_json=None, coco_override=None, sigclip_pipe=None):
    train_dir = os.path.join(dataset_path, "train")
    ann_path = os.path.join(train_dir, "_annotations.coco.json")
    # # readme_path = os.path.join(dataset_path, "README.roboflow.txt")
    # readme_path = os.path.join(dataset_path, "README.dataset.txt")
    readme_json_path = os.path.join("./data_instr/default", f"README.dataset_{os.path.basename(dataset_path)}.json")
    if not os.path.isfile(ann_path):
        print(f"No train annotations found in {train_dir}, skipping.")
        return None
    if few_shot_examples:
        # few_shot_dict = utils.build_few_shot_dict(dataset_path, examples_per_class=2)
        few_shot_dict = utils.build_few_shot_dict(dataset_path, examples_per_class=10)

        assert sum([len(v) for k,v in few_shot_dict.items()]) == len([f for f in os.listdir(train_dir) if f.split('.')[-1] in ['.png', 'jpeg', 'jpg']]), "Few-shot examples count does not match number of training images!"
    
        few_shot_samples = []
        [few_shot_samples.extend(v) for k,v in few_shot_dict.items() if len(v) > 0]
        print(f"Built few-shot examples dictionary with {len(few_shot_dict)} categories and {len(few_shot_samples)} total examples for {dataset_path}.")
    else:
        few_shot_dict = None
    
    dataset_instructions_json = {}
    if dataset_instructions_override_json is not None:
        dataset_instructions_json = dataset_instructions_override_json
    else:
        if os.path.isfile(readme_json_path):
            with open(readme_json_path, "r", encoding="utf-8") as f:
                dataset_instructions_json = json.load(f)
        print(f"Loaded dataset instructions for {dataset_path}: \n{dataset_instructions_json}\n")

    
    # coco_gt = COCO(ann_path)
    if coco_override is not None:
        coco_gt = coco_override
        print(f"Using coco_override for {dataset_path}.")
    else:
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

    # Check if all prediction files exist
    all_predictions_exist = (
        all(os.path.isfile(p) for p in prediction_cache_paths.values())
    )

    if all_predictions_exist and not args.ipt_mode:
        print(f"Using cached predictions for {dataset_path}")
        detections_all_by_type = {}
        for eval_type, path in prediction_cache_paths.items():
            with open(path, "r", encoding="utf-8") as f:
                detections_all_by_type[eval_type] = json.load(f)
    else:
        detections_all_by_type = {
            eval_type: [] for eval_type in eval_types
        }
        images_viz_count = 0
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
                img_id = img_info["id"]
                img_filename = img_info["file_name"]
                image_path = os.path.join(train_dir, img_filename)
    
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
                    # class_name_list=ds_cat_names, #GRG: Pass the entire list of category names
                    class_name_list=[eval_class_name],
                    no_instructions=no_instructions,
                    few_shot_dict=few_shot_dict,
                    output_dir=output_dir,
                    # eval_class_name=eval_class_name,
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
                
                # Yield results for live updates
                yield {
                    "img_id": img_id,
                    "image_path": image_path,
                    "gt_bboxes": [ann["bbox"] for ann in anns],
                    # "pred_bboxes": [det["bbox"] for det in all_detections["vqa_with_nms"]],
                    "pred_bboxes": [det["bbox"] for det in all_detections["vqa_no_nms"]],
                    "raw_output": raw_output,
                    "parsed_detections_before_nms": all_detections["vqa_no_nms"],
                    "parsed_detections": all_detections["vqa_with_nms"],
                    "all_detections": all_detections,
                    "gt_anns": anns,
                    "cat_dict": cat_dict,
                    "few_shot_examples_used": few_shot_examples_used,
                }

        del raw_output
        torch.cuda.empty_cache()
        gc.collect()
        
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

        if max_samples is not None:
            # --- Create a subset of coco_gt for evaluation ---
            processed_img_ids = {img['id'] for img in images_to_process}
            
            coco_gt_subset = COCO()
            coco_gt_subset.dataset['info'] = coco_gt.dataset.get('info', {})
            coco_gt_subset.dataset['licenses'] = coco_gt.dataset.get('licenses', [])
            coco_gt_subset.dataset['images'] = [img for img in coco_gt.dataset['images'] if img['id'] in processed_img_ids]
            coco_gt_subset.dataset['annotations'] = [ann for ann in coco_gt.dataset['annotations'] if ann['image_id'] in processed_img_ids]
            coco_gt_subset.dataset['categories'] = coco_gt.dataset['categories']
            coco_gt_subset.createIndex()
            # --- End of subset creation ---

            coco_dt = coco_gt_subset.loadRes(detections)
            coco_eval = COCOeval(coco_gt_subset, coco_dt, "bbox")
        else:
            coco_dt = coco_gt.loadRes(detections)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

        if eval_cat_id is not None:
            print(f"Evaluating on single category ID: {eval_cat_id}")
            coco_eval.params.catIds = [eval_cat_id]

        # coco_eval.params.imgIds = sorted(list(processed_img_ids))

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        all_stats[eval_type] = coco_eval.stats

    # Save the evaluation stats
    os.makedirs(eval_dir, exist_ok=True)
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



def generate_initial_class_definition(args, model, processor, class_name, initial_instructions, few_shot_examples):
    """
    Uses the VLM to generate an initial textual definition of a class based on all GT examples.
    """

    utils.set_seed(args.seed)

    if not few_shot_examples:
        return ""

    content = [
        {"type": "text", "text": 
       
        f"""
            Analyze the following images and describe the subjects or objects highlighted in green bounding boxes. 
            Identify and summarize the key visual characteristics that are consistently observed across these objects. 
            Emphasize the distinctive features that clearly differentiate this object class from other elements in the scene.

            Your goal is to produce a concise, clear, detailed, and generalizable definition that enables accurate recognition of this object class in future images and makes it easily distinguishable from other objects. 
            Do not mention bounding boxes, colors, or any annotation details in your response.
        """
        },
    ]
    
    # Concatenate images into a single message to avoid overwhelming the model
    # with too many separate image prompts if the list is very long.
    for example in few_shot_examples:
        # content.append({"type": "image", "image": example["image_path"]})
        content.append({"type": "image", "image": example})

    messages = [{"role": "user", "content": content}]
   
    definition, _ = utils.model_generate(messages, model, processor)

    # Clean up the definition
    definition = definition.replace(f"The visual characteristics of the '{class_name}' class are:", "").strip()
    definition = definition.replace(f"Definition of '{class_name}':", "").strip()
    
    print(f"Generated initial definition for '{class_name}': {definition}")
    return definition


def generate_class_definition(args, model, processor, class_name, current_instructions, few_shot_examples):
    """
    Uses the VLM to generate a textual definition of a class based on few-shot examples.
    """
    
    # utils.set_seed(args.seed)
    
    if not few_shot_examples:
        return ""

    content = [
        # {"type": "text", "text": f"Based on the following example images showing '{class_name}', describe the key visual characteristics of this class. Provide a concise definition that could be used to instruct someone on how to identify these objects. Do not mention the bounding boxes."},
        # {"type": "text", "text": f"Based on the following example images showing '{class_name}' in green bounding boxes, describe the key visual characteristics of this class. Provide a concise definition that could be used to instruct someone on how to identify these objects. Do not mention the bounding boxes."},
        {"type": "text", "text": 
        
        
        f"""Refine and improve the object class definition for '{class_name}' used in object detection: {current_instructions}

            Analyze the provided example images, where instances of the '{class_name}' class are shown with green bounding boxes, and enhance the definition to clearly describe its distinctive visual characteristics.

            Your refined definition should:
            - Capture the defining visual traits that distinguish this class from others.
            - Include objects similar to those shown with blue bounding boxes that visually align with the intended class examples.
            - Exclude objects similar to those shown with red bounding boxes that do not share the defining visual features of this class.

            Provide a concise, clear, and descriptive definition that could guide accurate object identification or annotation.
            Return only the updated definition for the '{class_name}' class.
            Do not mention bounding boxes, colors, or image annotations in your response.
        """

        },
    ]
    #TODO-GRG: Consider concatenating the images into a single image
    for example in few_shot_examples:
        content.append({"type": "image", "image": example["image_path"]})

    messages = [{"role": "user", "content": content}]
    
    definition, _ = utils.model_generate(messages, model, processor)
    print(f"Generated definition for '{class_name}': {definition}")
    
    return definition



def generate_class_definition_withFP(args, model, processor, class_name, current_instructions, correct_image, FP_error_image):
    """
    Uses the VLM to generate a textual definition of a class based on few-shot examples.
    """
    
    # utils.set_seed(args.seed)

    content = [
        # {"type": "text", "text": f"Based on the following example images showing '{class_name}', describe the key visual characteristics of this class. Provide a concise definition that could be used to instruct someone on how to identify these objects. Do not mention the bounding boxes."},
        # {"type": "text", "text": f"Based on the following example images showing '{class_name}' in green bounding boxes, describe the key visual characteristics of this class. Provide a concise definition that could be used to instruct someone on how to identify these objects. Do not mention the bounding boxes."},
        {"type": "text", "text": 
        


        f"""
            Analyze the image carefully and identify the key visual differences between the object shown in the green bounding box and the one shown in the red bounding box.

            Follow the following steps:
            Step-1. Describe the distinguishing visual characteristics that set apart the object in the green bounding box from the object in the red bounding box.
            Step-2. Based on these distinguishing traits, formulate a clear and descriptive class definition for the object in the green bounding box. This definition should focus on its unique visual and contextual features that help differentiate it from the object in the red bounding box.
            Step-3. Compare your new class definition with the existing definition of the '{class_name}' class provided below:
            
            Current class definition of the '{class_name}' class:
            \n{current_instructions}\n

            Step-4. Synthesize both definitions to produce an improved, more precise descriptive class definition for the '{class_name}' class. The updated definition should make it easier to accurately identify true instances of the '{class_name}' class while reducing false positives similar to the one seen in the red bounding box.

            Note: Do not mention bounding boxes, colors, or image annotations in your response. The updated class definition should be a textual description of the '{class_name}' class objects.

            Return the final updated class definition as descriptive text in the following format: ```python\n{{'{class_name}': <updated definition>}}\n```.

        """

        },
        # {"type": "text", "text": f"Here is the 'correct_image' showing the '{class_name}' class in green bounding boxes:"},
        # {"type": "image", "image": correct_image["image_path"]},
        {"type": "image", "image": correct_image},
        # {"type": "text", "text": f"Here is the 'FP_error_image' showing the false positive for the '{class_name}' class in red bounding boxes:"},
        # {"type": "image", "image": FP_error_image["image_path"]}
        {"type": "image", "image": FP_error_image}
    ]

    messages = [{"role": "user", "content": content}]
    
    definition, _ = utils.model_generate(messages, model, processor) 
    print(f"Generated definition for '{class_name}': {definition}")

    return definition



def generate_class_definition_withFN(args, model, processor, class_name, current_instructions, correct_image, FN_error_image):
    """
    Uses the VLM to generate a textual definition of a class based on few-shot examples.
    """
    
    # utils.set_seed(args.seed)

    content = [
        {"type": "text", "text": 

        f"""
            Analyze the image carefully and identify the key visual similarities between the object shown in the green bounding box and the one shown in the blue bounding box.

            Follow the following steps:
            Step-1. Describe the similar visual characteristics that set apart the object in the green bounding box from the object in the blue bounding box.
            Step-2. Based on these similarity traits, formulate a clear and descriptive class definition for the object in the blue bounding box as well as the object in the green bounding box. This definition should focus on its unique visual and contextual features that help identify both instances of the object in the green and blue bounding boxes.
            Step-3. Compare your new class definition with the existing definition of the '{class_name}' class provided below:

            Current class definition of the '{class_name}' class:
            \n{current_instructions}\n

            Step-4. Synthesize both definitions to produce an improved, more precise descriptive class definition for the '{class_name}' class. The updated definition should make it easier to accurately identify all true instances of the '{class_name}' class similar to the one seen in the blue and green bounding boxes.
            
            Note: Do not mention bounding boxes, colors, or image annotations in your response. The updated class definition should be a textual description of the '{class_name}' class objects.
            
            Return the final updated class definition as descriptive text in the following format: ```python\n{{'{class_name}': <updated definition>}}\n```.

        """

        },
        # {"type": "text", "text": f"Here is the 'correct_image' showing the '{class_name}' class in green bounding boxes:"},
        # {"type": "image", "image": correct_image["image_path"]},
        {"type": "image", "image": correct_image},
        # {"type": "text", "text": f"Here is the 'FP_error_image' showing the false positive for the '{class_name}' class in red bounding boxes:"},
        # {"type": "image", "image": FN_error_image["image_path"]}
        {"type": "image", "image": FN_error_image}
    ]

    messages = [{"role": "user", "content": content}]
    
    definition, _ = utils.model_generate(messages, model, processor)
    print(f"Generated definition for '{class_name}': {definition}")

    return definition


import re
import json
import ast

def extract_class_definition(response, class_name):
    """
    Extract all JSON/Python dict blocks from a given text.
    Works even if embedded in markdown-style ```python code fences.
    Returns a list of parsed Python dictionaries or raw strings (if parsing fails).
    """
    results = []

    # Step 1: Find all code blocks (optionally labeled as python or json)
    code_blocks = re.findall(r"```(?:python|json)?\s*(.*?)\s*```", response, re.DOTALL)
    
    for block in code_blocks:
        block = block.strip()
        parsed = None

        # Step 2: Try parsing as a Python dict first (since many are not strict JSON)
        try:
            parsed = ast.literal_eval(block)
        except Exception:
            # Step 3: If that fails, try JSON parsing
            try:
                parsed = json.loads(block)
            except Exception:
                parsed = block  # fallback: keep raw block text

        results.append(parsed)

    results_str = str(results)

    return results_str




def iterative_prompt_refinement(args, model, processor, dataset_path, num_iterations=3,
                                    num_samples=None, sigclip_pipe=None):
    
    """
    Performs iterative prompt refinement.
    1. Generates class definitions from few-shot examples.
    2. Evaluates the dataset.
    3. Identifies worst-performing examples.
    4. Refines the prompt and repeats.
    """

    utils.set_seed(args.seed)

    
    # Initial setup

    readme_json_path = os.path.join("./data_instr/default", f"README.dataset_{os.path.basename(dataset_path)}.json")
    class_instructions_json = {}
    if os.path.isfile(readme_json_path):
        with open(readme_json_path, "r", encoding="utf-8") as f:
            class_instructions_json = json.load(f)


    #Get all category ids
    train_dir = os.path.join(dataset_path, "train")
    ann_path = os.path.join(train_dir, "_annotations.coco.json")
    dataset_name = os.path.basename(dataset_path)
    coco_gt = COCO(ann_path)

    ds_cat_ids = coco_gt.getCatIds()
    ds_cat_names = [coco_gt.cats[cat_id]["name"] for cat_id in ds_cat_ids]
    #Dict of cat names to ids
    cat_name2id_dict = {coco_gt.cats[cat_id]["name"]: cat_id for cat_id in ds_cat_ids}

    #Dict of cat ids to names
    cat_dict = {cat_id: coco_gt.cats[cat_id]["name"] for cat_id in ds_cat_ids}
    print(f"Categories in {dataset_name}: {cat_dict}")

    if num_samples is not None:
        print(f"[Warning!] Limiting to {num_samples} samples per class for iterative prompt refinement.")

        # Ensure we don't request more samples than available
        # num_to_process = min(num_samples, len(coco_gt.dataset["images"]))
        
        # images_to_process = []
        processed_imgs = []
        for cat_id in ds_cat_ids:
            cat_name = coco_gt.cats[cat_id]["name"]

            # All images that have this category
            img_ids = coco_gt.getImgIds(catIds=[cat_id])
            

            # Randomly choose up to 3 images
            selected_img_ids = random.sample(img_ids, min(num_samples, len(img_ids)))
            print(f"Category '{cat_name}' ({cat_id}): Selected {len(selected_img_ids)} images out of {len(img_ids)} available.")
            
            # images_to_process.extend([coco_gt.loadImgs(img_id)[0] for img_id in selected_img_ids])
            processed_imgs.extend(selected_img_ids)
        
        # processed_img_ids = {img['id'] for img in processed_imgs}
        processed_img_ids = set(processed_imgs)
        print(f"Total unique images to process after sampling: {len(processed_img_ids)}")

        # --- Create a subset of coco_gt for evaluation ---
        
        coco_gt_subset = COCO()
        coco_gt_subset.dataset['info'] = coco_gt.dataset.get('info', {})
        coco_gt_subset.dataset['licenses'] = coco_gt.dataset.get('licenses', [])
        coco_gt_subset.dataset['images'] = [img for img in coco_gt.dataset['images'] if img['id'] in processed_img_ids]
        coco_gt_subset.dataset['annotations'] = [ann for ann in coco_gt.dataset['annotations'] if ann['image_id'] in processed_img_ids] # Also need to filter annotations to only those images
        coco_gt_subset.dataset['categories'] = coco_gt.dataset['categories']
        coco_gt_subset.createIndex()
        # --- End of subset creation ---

        coco_gt = coco_gt_subset


    #Create output directory for saving results
    result_dir = os.path.join(args.output_dir, "iterative_prompt_refinement")
    os.makedirs(result_dir, exist_ok=True)


    #Dataset results directory
    dataset_result_dir = os.path.join(result_dir, dataset_name)
    os.makedirs(dataset_result_dir, exist_ok=True)

    # --- Resume Logic ---
    all_iterm_refined_instructions_path = os.path.join(result_dir, f"all_iterm_refined_class_instructions_{dataset_name}.json")
    refined_class_instructions_json = {}
    if os.path.exists(all_iterm_refined_instructions_path):
        print(f"Found existing refined instructions file. Loading to resume: {all_iterm_refined_instructions_path}")
        try:
            with open(all_iterm_refined_instructions_path, "r", encoding="utf-8") as f:
                refined_class_instructions_json = json.load(f)
            print(f"Resuming. Loaded {len(refined_class_instructions_json)} completed class definitions.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read or parse existing instructions file. Starting from scratch. Error: {e}")
            refined_class_instructions_json = {}

    # --- Step 0: Generate initial class definitions from all GT examples ---
    
    for cat_id in ds_cat_ids: #GRG: This iterates over all categories in the dataset - which can is the right thing to do here - but can penalize results as we expect the model to predict all categories in each image
        class_name = coco_gt.cats[cat_id]["name"]
    
        if class_name in refined_class_instructions_json:
            print(f"Skipping class '{class_name}' as its definition already exists in the results file.")
            continue

        # --- Step 0: Generate class definition ---


        # --- Step 0: Generate class definition using positive samples only ---
        

        # Get all GT examples for this class

        # All images that have this category
        img_ids = coco_gt.getImgIds(catIds=[cat_id])
        if not img_ids or len(img_ids) == 0:
            # No images for this category
            raise ValueError(f"No images found for category '{class_name}' in the dataset.")

        # Randomly choose up to 3 images
        # selected_img_ids = random.sample(img_ids, min(examples_per_class, len(img_ids)))
        selected_img_ids = sorted(img_ids)
        
        gt_examples_for_class = []
        for chosen_img_id in selected_img_ids:
            ann_ids = coco_gt.getAnnIds(imgIds=[chosen_img_id], catIds=[cat_id])
            anns = coco_gt.loadAnns(ann_ids)

            img_info_list = coco_gt.loadImgs(chosen_img_id)
            if not img_info_list:
                continue

            img_info = img_info_list[0]
            image_path = os.path.join(train_dir, img_info["file_name"])
            if not os.path.isfile(image_path):
                continue

            # Visualize GT boxes for THIS category only
            gt_bboxes = [ann['bbox'] for ann in anns if ann['category_id'] == cat_id] #GRG: Only consider GT boxes for the current class

            if len(gt_bboxes) == 0:
                continue


            img = Image.open(image_path).convert("RGB")
            
            print(f"file_{os.path.basename(img_info['file_name'])} img.size: {img.size}")

            img_with_boxes = utils.draw_colored_bboxes_on_image(img, "green", gt_bboxes)

            img_viz_path = os.path.join(dataset_result_dir, f"few_shot_example_cls_{class_name}_initial_imId_{chosen_img_id}_file_{os.path.basename(img_info['file_name'])}.png")
            
            #Resize the image if too large
            max_dimension = (1920, 1080)  # Example max dimensions (width, height)
            img_with_boxes.thumbnail(max_dimension, Image.LANCZOS)
            print(f"file_{os.path.basename(img_info['file_name'])} resized img.size: {img_with_boxes.size}")
            
            #Save image
            img_with_boxes.save(img_viz_path)

            # gt_examples_for_class.append({"image_path": img_viz_path})
            gt_examples_for_class.append(img_with_boxes)



        if len(gt_examples_for_class) == 0:
            raise ValueError(f"No GT examples found for category '{class_name}' in the dataset.")

        def getInstructionsForClass(class_name, dataset_instructions_json):
            if class_name in dataset_instructions_json:
                dataset_instructions = dataset_instructions_json[class_name]
            else:
                # #Capitalize first letter to match keys
                # class_name_cap = class_name[0].upper() + class_name[1:]
                # dataset_instructions = dataset_instructions_json[class_name_cap]

                # Find the matching key ignoring case
                matched_key = next((key for key in dataset_instructions_json.keys() if key.lower() == class_name.lower()), None)
                if matched_key:
                    dataset_instructions = dataset_instructions_json[matched_key]
                else:
                    #Throw error
                    raise ValueError(f"Class name '{class_name}' not found in dataset instructions JSON keys.")

            return dataset_instructions

        # initial_instructions = class_instructions_json.get(class_name)
        initial_instructions = getInstructionsForClass(class_name, class_instructions_json)

        #Show initial instructions
        
        #Save initial instructions as text file
        org_instructions_path = os.path.join(dataset_result_dir, f"{class_name}_original_definition.txt")
        with open(org_instructions_path, "w", encoding="utf-8") as f:
            f.write(initial_instructions)

        # To avoid overwhelming the model, let's use a random sample of up to 10 examples for generation
        # examples_to_use = random.sample(gt_examples_for_class, min(10, len(gt_examples_for_class)))
        examples_to_use = gt_examples_for_class
        if len(gt_examples_for_class) != 10:
            print(f"Warning! GT examples count does not match expected number - 10!")
        

        #Check if initial definition with FP refinement already exists - then skip generation
        # init_def_path = os.path.join(dataset_result_dir, f"{class_name}_initial_definition_with_FP_{other_class_name}.txt")
        init_def_path = os.path.join(dataset_result_dir, f"{class_name}_initial_definition.txt")
        if os.path.exists(init_def_path):
            print(f"Found existing initial definition for '{class_name}'. Loading from: {init_def_path}")
            with open(init_def_path, "r", encoding="utf-8") as f:
                initial_instructions = f.read()
        else:
            # Generate the definition
            print(f"Generating initial definition for '{class_name}' using only GT examples.")

            # Generate the definition
            initial_instructions = generate_initial_class_definition(args, model, processor, class_name, initial_instructions, examples_to_use)
            
            # Save the generated initial instructions as text file
            init_def_path = os.path.join(dataset_result_dir, f"{class_name}_initial_with_only_gt_definition.txt")
            with open(init_def_path, "w", encoding="utf-8") as f:
                f.write(initial_instructions)
            


            # --- Step 0: Generate class definition using negative samples ---
            

            # Get negative examples from other classes
            for idx, other_cat_id in enumerate(ds_cat_ids):
                if other_cat_id == cat_id:
                    continue

                other_class_name = coco_gt.cats[other_cat_id]["name"]
                
                other_img_ids = coco_gt.getImgIds(catIds=[other_cat_id])

                if not other_img_ids or len(other_img_ids) == 0:
                    print(f"Warning! No images found for negative category '{other_class_name}' in the dataset.")
                    continue

                # Randomly choose 1 image
                chosen_other_img_id = random.choice(other_img_ids)
                
                other_ann_ids = coco_gt.getAnnIds(imgIds=[chosen_other_img_id], catIds=[other_cat_id])
                other_anns = coco_gt.loadAnns(other_ann_ids)

                other_img_info_list = coco_gt.loadImgs(chosen_other_img_id)
                if not other_img_info_list:
                    continue

                other_img_info = other_img_info_list[0]
                other_image_path = os.path.join(train_dir, other_img_info["file_name"])
                if not os.path.isfile(other_image_path):
                    continue

                # Visualize GT boxes for THIS category only
                other_gt_bboxes = [ann['bbox'] for ann in other_anns if ann['category_id'] == other_cat_id] #GRG: Only consider GT boxes for the current class

                if len(other_gt_bboxes) == 0:
                    continue


                other_img = Image.open(other_image_path).convert("RGB")
                
                other_img_with_boxes = utils.draw_colored_bboxes_on_image(other_img, "red", other_gt_bboxes)


                #Resize the image if too large
                max_dimension = (1920, 1080)  # Example max dimensions (width, height)
                other_img_with_boxes.thumbnail(max_dimension, Image.LANCZOS)
                print(f"file_{os.path.basename(other_img_info['file_name'])} resized img.size: {other_img_with_boxes.size}")
                

                other_img_viz_path = os.path.join(dataset_result_dir, f"few_shot_example_cls_{class_name}_initial_other_imId_{chosen_other_img_id}_file_{os.path.basename(other_img_info['file_name'])}.png")
                #Save image
                other_img_with_boxes.save(other_img_viz_path)

                # fp_examples_for_class = {"image_path": other_img_viz_path}
                fp_examples_for_class = other_img_with_boxes

                # positive_examples_for_class = random.choice(examples_to_use)
                if len(examples_to_use) > idx:
                    positive_examples_for_class = examples_to_use[idx]
                else:
                    positive_examples_for_class = random.choice(examples_to_use)


                # --- Refine prompt for next iteration ---
                
                #False-positive focused refinement
                fp_generated_definition_analysis = generate_class_definition_withFP(args, model, processor, class_name, initial_instructions, positive_examples_for_class, fp_examples_for_class)
            
                fp_generated_definition = extract_class_definition(fp_generated_definition_analysis, class_name)

                if fp_generated_definition:
                    initial_instructions = fp_generated_definition


                    # Save the generated initial instructions as text file
                    init_def_path = os.path.join(dataset_result_dir, f"{class_name}_initial_definition_with_FP_{other_class_name}.txt")
                    with open(init_def_path, "w", encoding="utf-8") as f:
                        f.write(initial_instructions)
                            
            


            # Save the generated initial instructions as text file
            init_def_path = os.path.join(dataset_result_dir, f"{class_name}_initial_definition.txt")
            with open(init_def_path, "w", encoding="utf-8") as f:
                f.write(initial_instructions)


        current_instructions = initial_instructions
        prev_instructions = initial_instructions

        best_mAP = -1.0
        best_instructions = initial_instructions
        prev_mAP = -1.0
        
        prev_worst_examples_map = {}  # To store worst-performing examples from previous iteration
        
        # --- Iteration-level Resume Logic ---
        iteration_state_path = os.path.join(dataset_result_dir, f"ipt_state_{dataset_name}_{class_name}.json")
        start_iteration = 0
        instruction_refinements = {}
        
        if os.path.exists(iteration_state_path):
            print(f"Found existing iteration state file for class '{class_name}'. Loading to resume.")
            try:
                with open(iteration_state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                start_iteration = state.get("last_completed_iteration", -1) + 1
                current_instructions = state.get("current_instructions", initial_instructions)
                best_instructions = state.get("best_instructions", initial_instructions)
                best_mAP = state.get("best_mAP", -1.0)
                prev_mAP = state.get("prev_mAP", -1.0)
                prev_instructions = state.get("prev_instructions", initial_instructions)
                instruction_refinements = state.get("instruction_refinements", {})
                print(f"Resuming from iteration {start_iteration} for class '{class_name}'.")
            except (json.JSONDecodeError, IOError, KeyError) as e:
                print(f"Warning: Could not read or parse iteration state file. Starting from iteration 0. Error: {e}")
                start_iteration = 0
                instruction_refinements = {}

        if start_iteration >= num_iterations:
            print(f"All {num_iterations} iterations already completed for class '{class_name}'. Skipping.")
            continue

        for i in range(start_iteration, num_iterations):

            print(f"\n\n\n--- Iteration {i} for class '{class_name}' [{ds_cat_ids.index(cat_id)+1}/{len(ds_cat_ids)}] ---\n\n\n")

            # stats_type = "vqa_with_nms"
            stats_type = "vqa_no_nms"
            # stats_type = "orig_with_nms"


            # --- Step 2: Evaluate with the current prompt ---
            # run_name = f"ipt_iter_{i}"
            run_name = f"class_{class_name}_ipt_iter_{i}"
            
            # UI placeholders for live visualization
            
            # Get the number of samples for the progress bar
            # temp_coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
            # num_eval_samples = len(temp_coco.dataset["images"])
            # del temp_coco
            num_eval_samples = len(coco_gt.dataset["images"])

            current_instructions_json = {class_name: current_instructions}

            #Get current seed state - to restore after evaluation - as evaluation changes it
            seed_state = get_seed_state()

            eval_generator = evaluate_dataset(
                args, model, processor, dataset_path,
                no_instructions=False,  # We are using generated instructions
                few_shot_examples=False, # Few-shot examples were used for definition generation
                run_name=run_name,
                output_dir=dataset_result_dir,
                # dataset_instructions_override_json=current_instructions,
                dataset_instructions_override_json=current_instructions_json,
                eval_class_name=class_name,
                eval_cat_id=cat_id, #GRG: Pass the cat_id for evaluation
                # max_samples=None  # Evaluate on the full dataset to get proper metrics
                # max_samples=5 #10 #2 #8 #5  #TODO-GRG: Need to remove - For testing purposes only
                coco_override=coco_gt if num_samples is not None else None, # Pass the coco_gt with limited samples if applicable
                sigclip_pipe=sigclip_pipe
            )

            #Restore seed state
            set_seed_from_state(seed_state)
            
            all_results_for_iter = []
            for j, result in enumerate(eval_generator):
                all_results_for_iter.append(result)
                

            # --- Display mAP and AR ---

            eval_dir = os.path.join(dataset_result_dir, "evaluations", run_name)
            eval_results_path = os.path.join(eval_dir, f"evaluation_{dataset_name}.json")
            if os.path.exists(eval_results_path):
                with open(eval_results_path, 'r') as f:
                    all_stats_dict = json.load(f)
                
                # Display the primary metrics
                # ap50_95 = all_stats_dict.get("vqa_with_nms", [0.0]*12)[0]
                ap50_95 = all_stats_dict.get(stats_type, [0.0]*12)[0]
                # ar100 = all_stats_dict.get("vqa_with_nms", [0.0]*12)[8]
                ar1 = all_stats_dict.get(stats_type, [0.0]*12)[6]
                
                print(f"Iteration {i} - Class '{class_name}': mAP@.50-.95 = {ap50_95:.4f}, AR@1 = {ar1:.4f}")

                instruction_refinements[f"class_{class_name}_iter_{i}"] = {
                    "mAP_50_95": ap50_95,
                    "AR_1": ar1,
                    "instructions": current_instructions
                }

                # if ap50_95 > best_mAP:
                if ap50_95 >= best_mAP:
                    best_mAP = ap50_95
                    best_instructions = current_instructions

    
                if ap50_95 < prev_mAP:
                    print(f"Iteration {i} - Class '{class_name}': mAP decreased from previous iteration ({prev_mAP:.4f} to {ap50_95:.4f}). Reverting to previous instructions.")
                    current_instructions = prev_instructions
                    # continue  # Skip to next class without refining
                else:
                    prev_instructions = current_instructions
                    prev_mAP = ap50_95
                


            # stats_type = "orig_with_nms"

            # --- Step 3: Identify worst-performing examples (simplified) ---
            # A simple heuristic: find images with the most false negatives (missed GT objects).

            def get_other_cls_ious(other_cls_gt_bboxes, pred_box):

                #If no other class GT boxes, return zero
                if len(other_cls_gt_bboxes) == 0:
                    return 0.0
                
                else:
                    # Calculate IoU for all pairs
                    other_cls_iou = [utils.calculate_iou(gt_box, pred_box) for gt_box in other_cls_gt_bboxes]

                    # Find best match 
                    best_other_cls_ious = max(other_cls_iou)

                    return best_other_cls_ious



            image_pred_performance = []
            for result in all_results_for_iter:

                gt_bboxes = [ann['bbox'] for ann in result['gt_anns'] if ann['category_id'] == cat_id] #GRG: Only consider GT boxes for the current class
                
                other_cls_gt_bboxes = [ann['bbox'] for ann in result['gt_anns'] if ann['category_id'] != cat_id] #GRG: GT boxes for other classes
                
                
                # pred_detections = result['all_detections']['vqa_no_nms']
                pred_detections = result['all_detections'][stats_type] #GRG: Use the same stats type as evaluation


                #No predictions - check if there are GT boxes for false negatives
                if len(pred_detections) == 0:

                    
                    # If there are GT boxes but no predictions, all are false negatives
                    if len(gt_bboxes) > 0:
                        
                        for gt_bbox in gt_bboxes:
                            
                            # fn_error = 1.0
                            det_score = 0.0
                            gt_iou = 0.0
                            best_score = 0.0
                            other_cls_iou = 0.0
                            fp_error = 0.0

                            image_pred_performance.append({
                                    "img_id": result['img_id'], 
                                    "gt_iou": gt_iou,
                                    "best_score": best_score, 
                                    # "fn_error": fn_error,
                                    "other_cls_iou": other_cls_iou,
                                    "fp_error": fp_error,
                                    "image_path": result['image_path'],
                                    "gt_bbox": gt_bbox,
                                    "pred_bbox": [],
                                    "det_score": det_score,
                                    "det": None,
                                })
                            
                    continue  


                for det in pred_detections:

                    if len(gt_bboxes) == 0:
                        # All predictions are false positives
                        det_score = det['score']
                        pred_box = det['bbox']
                        other_cls_iou = get_other_cls_ious(other_cls_gt_bboxes, pred_box)
                        # fp_error = det_score * other_cls_iou
                        fp_error = det_score * max(0.2, other_cls_iou)
                        gt_iou = 0.0
                        # best_score = 0.0
                        best_score = -1.0
                        # fn_error = 0.0
                        
                        
                        image_pred_performance.append({
                                "img_id": result['img_id'], 
                                "gt_iou": gt_iou,
                                "best_score": best_score, 
                                # "fn_error": fn_error,
                                "other_cls_iou": other_cls_iou,
                                "fp_error": fp_error,
                                "image_path": result['image_path'],
                                "gt_bbox": None,
                                "pred_bbox": pred_box,
                                "det_score": det_score,
                                "det": det,
                            })
                        continue

                    
                    #Normal case: There are both GT boxes and predictions
                

                    # Calculate metrics for each GT box
                    det_score = det['score']
                    pred_box = det['bbox']

                    gt_iou_list = [utils.calculate_iou(gt_box, pred_box) for gt_box in gt_bboxes]
                    gt_bbox = gt_bboxes[np.argmax(gt_iou_list)] #Best matching GT box
                    gt_iou = max(gt_iou_list) #Best matching GT box IoU
                    
                    best_score = det_score * gt_iou
                    # fn_error = (1 - det_score) * (1 - gt_iou) #TODO-GRG: Consider det_score in FN error?
                    # fn_error = 1 - gt_iou 
                    other_cls_iou = get_other_cls_ious(other_cls_gt_bboxes, pred_box)
                    
                    if gt_iou > 0.0:
                        #If this prediction matches a GT box, it cannot be a FP
                        fp_error = 0.0
                    else:
                        # fp_error = det_score * other_cls_iou #We only consider FP error if it matches other class GT boxes - as we can have true det without gt labels as this is few-shot setting
                        fp_error = det_score * max(0.2, other_cls_iou)

                    image_pred_performance.append({
                            "img_id": result['img_id'], 
                            "gt_iou": gt_iou,
                            "best_score": best_score, 
                            # "fn_error": fn_error,
                            "other_cls_iou": other_cls_iou,
                            "fp_error": fp_error,
                            "image_path": result['image_path'],
                            "gt_bbox": gt_bbox,
                            "pred_bbox": pred_box,
                            "det_score": det_score,
                            "det": det,
                        })
                    
                

            # Select one best match, one worst FP, and one worst FN
            # Randomly choose from the top 3 for each category to introduce diversity
            top_n = 5 #3
            best_match_candidates = sorted(image_pred_performance, key=lambda x: x['best_score'], reverse=True)[:top_n]
            # best_match_candidates = sorted(image_pred_performance, key=lambda x: x['best_score'], reverse=True)
            best_match_candidates = [c for c in best_match_candidates if c['best_score'] > 0.0] # Only consider truly good matches
            
            #Exclude previous best match image if multiple candidates exist
            if len(best_match_candidates) > 1:
                best_match_candidates_filterd = [c for c in best_match_candidates if c['img_id'] != prev_worst_examples_map['best_match']['img_id']] if 'best_match' in prev_worst_examples_map and prev_worst_examples_map['best_match'] is not None else best_match_candidates
                if len(best_match_candidates_filterd) > 0:
                    best_match_candidates = best_match_candidates_filterd

            worst_fp_candidates = sorted(image_pred_performance, key=lambda x: x['fp_error'], reverse=True)[:top_n]
            # worst_fp_candidates = sorted(image_pred_performance, key=lambda x: x['fp_error'], reverse=True)
            worst_fp_candidates = [c for c in worst_fp_candidates if c['fp_error'] > 0.0] # Only consider truly good matches
            
            #Exclude previous worst FP if multiple candidates exist
            if len(worst_fp_candidates) > 1:
                worst_fp_candidates_filterd = [c for c in worst_fp_candidates if c['img_id'] != prev_worst_examples_map['worst_fp']['img_id']] if 'worst_fp' in prev_worst_examples_map and prev_worst_examples_map['worst_fp'] is not None else worst_fp_candidates
                if len(worst_fp_candidates_filterd) > 0:
                    worst_fp_candidates = worst_fp_candidates_filterd

            # worst_fn_candidates = sorted(image_pred_performance, key=lambda x: x['fn_error'], reverse=True)[:top_n]
            worst_fn_candidates_dict = {}
            for c in image_pred_performance:
                # imgs with no GT boxes cannot have false negatives
                if c['best_score'] == -1.0: 
                    continue

                #Add FN error 
                fn_error = 1.0 - c['best_score']
                c['fn_error'] = fn_error

                key = f"{c['img_id']}_{c['gt_bbox']}"

                if key not in worst_fn_candidates_dict:
                    worst_fn_candidates_dict[key] = c
                else:
                    # Keep the one with higher best score - as this is the matching pred_bbox with gt_box that FN needs to be considered against
                    if c['best_score'] > worst_fn_candidates_dict[key]['best_score']:
                        worst_fn_candidates_dict[key] = c

            worst_fn_candidates = sorted(worst_fn_candidates_dict.values(), key=lambda x: x['fn_error'], reverse=True)[:top_n]
            # worst_fn_candidates = sorted(worst_fn_candidates_dict.values(), key=lambda x: x['fn_error'], reverse=True)
            worst_fn_candidates = [c for c in worst_fn_candidates if c['fn_error'] > 0.0] # Only consider truly good matches

            #Exclude previous worst FN if multiple candidates exist
            if len(worst_fn_candidates) > 1:
                worst_fn_candidates_filterd = [c for c in worst_fn_candidates if c['img_id'] != prev_worst_examples_map['worst_fn']['img_id']] if 'worst_fn' in prev_worst_examples_map and prev_worst_examples_map['worst_fn'] is not None else worst_fn_candidates
                if len(worst_fn_candidates_filterd) > 0:
                    worst_fn_candidates = worst_fn_candidates_filterd

            best_match_example = random.choice(best_match_candidates) if len(best_match_candidates) > 0 else None
            worst_fp_example = random.choice(worst_fp_candidates) if len(worst_fp_candidates) > 0 else None
            worst_fn_example = random.choice(worst_fn_candidates) if len(worst_fn_candidates) > 0 else None

            # Combine them, ensuring uniqueness
            worst_examples_map = {
                'best_match': best_match_example,
                'worst_fp': worst_fp_example,
                'worst_fn': worst_fn_example
            }

            prev_worst_examples_map = worst_examples_map
            
            print(f"Worst examples selected for iteration {i+1}, class '{class_name}': \n{worst_examples_map}")
                     
            few_shot_examples = {}
            for idx, (ex_type, ex) in enumerate(worst_examples_map.items()):
                if ex is None:
                    print(f"No example found for {ex_type} in iteration {i+1} for class '{class_name}'.")
                    continue

                img = Image.open(ex['image_path']).convert("RGB")
                if ex_type == 'best_match':
                    # if not ex['gt_bbox']: continue
                    img_with_boxes = utils.draw_colored_bboxes_on_image(img, "green", [ex['gt_bbox']])
                    caption = f"Best Match (score: {ex['best_score']:.2f})"
                
                elif ex_type == 'worst_fp':
                    # if not ex['pred_bbox']: continue
                    #TODO-GRG: We need to ensure that there aren't any pred boxes that match GT boxes here
                    #TODO-GRG: We also need to ensure that there aren't any pred boxes that are actually right but shown wrong as the gt label is not there due to few-shot

                    img_with_boxes = utils.draw_colored_bboxes_on_image(img, "red", [ex['pred_bbox']])
                    caption = f"Worst FP (Error: {ex['fp_error']:.2f})"
                
                elif ex_type == 'worst_fn':
                    # if not ex['gt_bbox']: continue
                    img_with_boxes = utils.draw_colored_bboxes_on_image(img, "blue", [ex['gt_bbox']])
                    caption = f"Worst FN (Error: {ex['fn_error']:.2f})"
                
                else:
                    print(f"[Warning] Unknown example type: {ex_type} for iteration {i+1}, class '{class_name}' with example: {ex}")
                    continue


                
                #Resize the image if too large
                max_dimension = (1920, 1080)  # Example max dimensions (width, height)
                img_with_boxes.thumbnail(max_dimension, Image.LANCZOS)
                print(f"file_{os.path.basename(ex['image_path'])} resized img.size [org size = {img.size}]: {img_with_boxes.size}")
                
                img_viz_path = os.path.join(dataset_result_dir, f"few_shot_example_cls_{class_name}_iter{i}_{ex_type}_imId_{ex['img_id']}_caption_{caption}.png")
                #Save image
                img_with_boxes.save(img_viz_path)

                # few_shot_examples[ex_type] = {"image_path": img_viz_path}
                few_shot_examples[ex_type] = img_with_boxes




            # --- Step 4: Refine prompt for next iteration ---
            
            # generated_definition = generate_class_definition(args, model, processor, class_name, current_instructions, few_shot_examples)
            
            # Save the analysis text to a file
            analysis_path = os.path.join(dataset_result_dir, f"{class_name}_analysis_iter_{i}.txt")
            with open(analysis_path, "w", encoding="utf-8") as f:
                f.write(f"Analysis for class '{class_name}' at iteration {i+1}:\n\n")
                f.write(f"Current Instructions:\n{current_instructions}\n\n")

                

            fn_generated_definition, fp_generated_definition = None, None
            #False-negative focused refinement
            if 'best_match' in few_shot_examples and 'worst_fn' in few_shot_examples:
                fn_generated_definition_analysis = generate_class_definition_withFN(args, model, processor, class_name, current_instructions, few_shot_examples['best_match'], few_shot_examples['worst_fn'])
            
                fn_generated_definition = extract_class_definition(fn_generated_definition_analysis, class_name)

                if fn_generated_definition:
                    # prev_instructions = current_instructions
                    current_instructions = fn_generated_definition
                    
                    with open(analysis_path, "a", encoding="utf-8") as f:
                        f.write(f"Generated Analysis for False-Negative based Class Definition:\n{fn_generated_definition_analysis}\n\n")

            #False-positive focused refinement
            if 'best_match' in few_shot_examples and 'worst_fp' in few_shot_examples:
                fp_generated_definition_analysis = generate_class_definition_withFP(args, model, processor, class_name, current_instructions, few_shot_examples['best_match'], few_shot_examples['worst_fp'])
            
                fp_generated_definition = extract_class_definition(fp_generated_definition_analysis, class_name)

                if fp_generated_definition:
                    # if not fn_generated_definition: #Only update prev_instructions if FN refinement was not done
                    #     prev_instructions = current_instructions
                    current_instructions = fp_generated_definition

                    with open(analysis_path, "a", encoding="utf-8") as f:
                        f.write(f"Generated Analysis for False-Positive based Class Definition:\n{fp_generated_definition_analysis}\n\n")
                        f.write(f"Refined Instructions:\n{current_instructions}\n")

            
            # Display the analysis
            
             # --- Iteration-level Save for Resume ---
            print(f"Finished iteration {i} for class '{class_name}'. Saving state.")
            iteration_state = {
                "last_completed_iteration": i,
                "current_instructions": current_instructions,
                "best_instructions": best_instructions,
                "best_mAP": best_mAP,
                "prev_mAP": prev_mAP,
                "prev_instructions": prev_instructions,
                "instruction_refinements": instruction_refinements,
            }
            with open(iteration_state_path, "w", encoding="utf-8") as f:
                json.dump(iteration_state, f, indent=2)
            print(f"Saved iteration state to {iteration_state_path}")

            


        #Display Initial Instructions
        
        print(f"Initial instructions: \n{initial_instructions}")

        #Display Final Refined Instructions

        
        print(f"Final refined instructions: \n{current_instructions}")

        #Save final refined instructions
        refined_instructions_path = os.path.join(dataset_result_dir, f"refined_instructions_{dataset_name}_cls_{class_name}.txt")
        with open(refined_instructions_path, "w", encoding="utf-8") as f:
            f.write(current_instructions)
        print(f"Saved final refined instructions to {refined_instructions_path}")

        instruction_refinements["final_refined_instructions"] = current_instructions


        print(f"Best instructions achieved during iterations (mAP: {best_mAP:.4f}): \n{best_instructions}")

        best_instructions_path = os.path.join(dataset_result_dir, f"best_instructions_{dataset_name}_cls_{class_name}.txt")
        with open(best_instructions_path, "w", encoding="utf-8") as f:
            f.write(best_instructions)
        print(f"Saved best instructions to {best_instructions_path}")

        instruction_refinements["best_instructions"] = best_instructions


        #Save instruction refinements log
        instruction_refinements_log_path = os.path.join(dataset_result_dir, f"instruction_refinements_log_{dataset_name}_cls_{class_name}.json")
        with open(instruction_refinements_log_path, "w", encoding="utf-8") as f:
            json.dump(instruction_refinements, f, indent=2)
        print(f"Saved instruction refinements log to {instruction_refinements_log_path}")


        refined_class_instructions_json[class_name] = best_instructions

        # --- Incremental Save for Resume ---
        print(f"\nFinished processing class '{class_name}'. Saving intermediate results to allow for resuming.")
        with open(all_iterm_refined_instructions_path, "w", encoding="utf-8") as f:
            json.dump(refined_class_instructions_json, f, indent=2)
        print(f"Saved intermediate refined instructions to {all_iterm_refined_instructions_path}\n")


    # Save all refined class instructions
    all_refined_instructions_path = os.path.join(result_dir, f"all_refined_class_instructions_{dataset_name}.json")
    with open(all_refined_instructions_path, "w", encoding="utf-8") as f:
        json.dump(refined_class_instructions_json, f, indent=2)
    print(f"Saved all refined class instructions to {all_refined_instructions_path}")


    # return current_instructions
    return refined_class_instructions_json



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

    
    # Run the iterative prompt refinement process
    if os.path.exists(os.path.join(args.output_dir, "iterative_prompt_refinement", f"all_refined_class_instructions_{os.path.basename(dataset_path)}.json")):
        print(f"Refined class instructions already exist. Skipping IPT.")
        
    else:
        dataset_instructions_override_json = iterative_prompt_refinement(
            args,
            model=model,
            processor=processor,
            dataset_path=dataset_path,
            num_iterations=args.num_ipt_iterations,
            sigclip_pipe=sigclip_pipe if args.siglip_rescore else None
        )


    # After use free up memory:
    del model
    torch.cuda.empty_cache()
    gc.collect()


    print("\n\n" + "*" * 60 + "\n")
    print(f"Starting the final evaluation with the new refined class definitions...")
    print("\n" + "*" * 60 + "\n\n")

    args.data_instr_path = os.path.join(args.output_dir, "iterative_prompt_refinement", f"all_refined_class_instructions")
    args.output_dir = os.path.join(args.output_dir, f"final_instruction_eval")
    evaluator.run_single_dataset_evaluation(args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen3-VL-235B-A22B-Instruct", help='model name e.g., Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-72B-Instruct, Qwen3-VL-8B-Instruct, Qwen3-VL-30B-A3B-Instruct, Qwen3-VL-235B-A22B-Instruct]')
    parser.add_argument("--no_instructions", action="store_true", help="Run inference with no instructions")
    parser.add_argument("--few_shot", action="store_true", help="Use 3 random few-shot examples from test set")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to a single dataset to evaluate. If not set, all datasets will be evaluated in parallel.")
    parser.add_argument("--output_dir", type=str, default="results/rf100vl_IPT/rf20_IPT_singleclass_vqaScore_withNMS", help="Directory to save results and visuals.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None, help='List of GPU IDs to use for processing. e.g. --gpu_ids 0 1 4')
    parser.add_argument('--vqa_batch_size', type=int, default=8, help='Batch size for VQA scoring of candidate masks.')
    parser.add_argument("--vqa_rescore", action="store_true", help="Use VQA-based re-scoring of candidate masks")
    parser.add_argument("--siglip_rescore", action="store_true", help="Use SigLip-based re-scoring of candidate masks")
    parser.add_argument("--apply_nms", action="store_true", help="Apply Non-Maximum Suppression to detections.")
    parser.add_argument("--nms_threshold", type=float, default=0.5, help="IoU threshold for Non-Maximum Suppression.")
    parser.add_argument("--class_rescore", action="store_true", help="Use VQA-based class re-scoring of candidate masks")
    parser.add_argument("--ipt_mode", action="store_true", help="Enable Iterative Prompt Tuning (requires --dataset_path).")
    parser.add_argument("--num_ipt_iterations", type=int, default=3, help="Number of iterations for IPT.")

    parser.add_argument("--device_map_auto", action="store_true", help="Use device_map='auto' for model loading. Overrides --qwen_device if set.")

    args = parser.parse_args()

    run_single_dataset_evaluation(args)