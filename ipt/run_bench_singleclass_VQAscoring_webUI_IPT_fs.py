'''
Run cmd: CUDA_VISIBLE_DEVICES=0 python code/rf100vl/qwen-2.5-vl-rf-fsod-master/run_bench_singleclass_VQAscoring_webUI.py --eval --dataset_path ../rf100-vl/ --vqa_rescore --no_instructions
Run cmd: CUDA_VISIBLE_DEVICES=0 python code/rf100vl/qwen-2.5-vl-rf-fsod-master/run_bench_singleclass_VQAscoring_webUI.py --eval --vqa_rescore --no_instructions --output_dir results/rf100vl/rf20_singleclass_codePrompt_vqaScore_v1 --gpu_ids 0 1 2 3 4 5 6 7
Run cmd: CUDA_VISIBLE_DEVICES=0 python code/rf100vl/qwen-2.5-vl-rf-fsod-master/run_bench_singleclass_VQAscoring_webUI.py --eval --vqa_rescore --class_rescore --no_instructions --output_dir results/rf100vl/rf20_singleclass_codePrompt_vqaScore_v1 --gpu_ids 0 1 2 3 4 5 6 7
Run cmd: CUDA_VISIBLE_DEVICES=0 python code/rf100vl/qwen-2.5-vl-rf-fsod-master/run_bench_singleclass_VQAscoring_webUI.py --eval --vqa_rescore --class_rescore --no_instructions --apply_nms --nms_threshold 0.5 --output_dir results/rf100vl/rf20_singleclass_codePrompt_vqaScore_v1 --gpu_ids 0 1 2 3 4 5 6 7
Run cmd: CUDA_VISIBLE_DEVICES=0 python code/rf100vl/qwen-2.5-vl-rf-fsod-master/run_bench_singleclass_VQAscoring_webUI.py --eval --vqa_rescore --class_rescore --no_instructions --apply_nms --nms_threshold 0.5 --dataset_path wb-prova --output_dir results/rf100vl_new/rf20_singleclass_codePrompt_vqaScore_classRescore_nms0.5_v1 --gpu_ids 0 1 2 3 4 5 6 7
Run cmd: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python code/rf100vl/qwen-2.5-vl-rf-fsod-master/run_bench_singleclass_VQAscoring_webUI.py --eval --vqa_rescore --few_shot --apply_nms --nms_threshold 0.5 --dataset_path wb-prova --output_dir results/rf100vl_fixedPadBug/rf20_singleclass_codePrompt_vqaScore_nms0.5_fewShot_v1 --device_map_auto
'''

import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import glob
import torch
import shutil
from PIL import Image
from tqdm import tqdm
import streamlit as st

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import time
import gc
from PIL import ImageDraw, ImageFilter, ImageFont
import numpy as np
import argparse
import re
import random
import pandas as pd
import subprocess

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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


def _load_qwen_model_raw(qwen_device="cuda:0", device_map_auto=False):
    """Loads the Qwen model and processor."""
    device_map_config = "auto" if device_map_auto else {"": qwen_device}
    print(f"Loading Qwen model with device_map: {device_map_config}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        # torch_dtype="auto",
        torch_dtype= torch.bfloat16,
        attn_implementation="flash_attention_2",
        # device_map="auto"
        # device_map={"": qwen_device}
        device_map=device_map_config
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    model.eval()


    print("processor.tokenizer.padding_side:", processor.tokenizer.padding_side)
    print("processor.tokenizer.pad_token:", processor.tokenizer.pad_token)
    print("processor.tokenizer.eos_token:", processor.tokenizer.eos_token)

    # ✅ Fix padding side and pad token
    tokenizer = processor.tokenizer  # access the underlying tokenizer

    tokenizer.padding_side = "left"  # left padding for decoder-only models


    # 2. Keep "<|endoftext|>" as pad token (already set)
    # Do NOT overwrite with eos_token ("<|im_end|>")

    # # Ensure PAD token exists
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    # Save these updates into the processor so it uses them
    processor.tokenizer = tokenizer

    print("processor.tokenizer.padding_side:", processor.tokenizer.padding_side)
    print("processor.tokenizer.pad_token:", processor.tokenizer.pad_token)
    print("processor.tokenizer.eos_token:", processor.tokenizer.eos_token)


    # Optionally, if you want to persist this behavior for future loads:
    # processor.save_pretrained("./qwen2_5_vl_leftpad")

    return model, processor

@st.cache_resource
def load_qwen_model_cached(qwen_device="cuda:0"): # For Streamlit
    """Cached loader for the Qwen model and processor for Streamlit. Note: device_map_auto is not supported in Streamlit mode via this function."""
    return _load_qwen_model_raw(qwen_device, device_map_auto=False)

def load_qwen_model(qwen_device="cuda:0", device_map_auto=False): # For CLI
    """Loads the Qwen model and processor for CLI use."""
    return _load_qwen_model_raw(qwen_device, device_map_auto)

def draw_bboxes_on_image(image, pred_bboxes, gt_bboxes):
    """
    Draws predicted bboxes (red) and ground-truth bboxes (green) on the image.
    Returns a new PIL image with the boxes drawn.
    Bboxes should be [x, y, w, h] in image coordinates.
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    # GT in green
    for (x, y, w, h) in gt_bboxes:
        draw.rectangle([(x, y), (x + w, y + h)], outline="green", width=8)

    # Pred in red
    for (x, y, w, h) in pred_bboxes:
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=8)

    return img_copy



def draw_colored_bboxes_on_image(image, color, bboxes):
    """
    Draws colored bboxes on the image.
    Returns a new PIL image with the boxes drawn.
    Bboxes should be [x, y, w, h] in image coordinates.
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    for (x, y, w, h) in bboxes:
        draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=8)

    return img_copy



def draw_bboxes_with_labels_on_image(image, pred_detections, gt_anns, cat_id2name_dict, font_size=None):
    """
    Draws predicted bboxes (red) and ground-truth bboxes (green) on the image.
    Returns a new PIL image with the boxes drawn.
    Bboxes should be [x, y, w, h] in image coordinates.
    """
    img_copy = image.copy()

    if font_size is None:
        # Dynamically set font size based on image height
        font_size = max(15, int(image.height * 0.04))

    draw = ImageDraw.Draw(img_copy)
    
    try:
        # font_path = "arial.ttf"
        font_path = "DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Warning: Font '{font_path}' not found. Loading default font.")
        # The default font does not support a size parameter, so we can't resize it.
        # The text will be small if the default font is used.
        font = ImageFont.load_default() 

    # GT in green
    # for (x, y, w, h) in gt_bboxes:
    for ann in gt_anns:
        bbox = ann["bbox"]
        gt_label = cat_id2name_dict[ann["category_id"]]
        
        x, y, w, h = bbox
        draw.rectangle([(x, y), (x + w, y + h)], outline="green", width=8)

        # Add label text for GT
        # text_position = (x, y + font_size*0.01 + 2) if y > (font_size*0.01 + 20) else (x, y - h - 20)
        # text_position = (x, y - font_size*0.01 - 50)
        text_position = (x, y - font_size*1.1)
        # text_position = (x, y - 20) if y > 20 else (x, y + 20)
        draw.text(text_position, gt_label, fill="green", font=font)
        
    # Pred in red
    # for (x, y, w, h) in pred_bboxes:
    for det in pred_detections:
        bbox = det["bbox"]
        pred_label = det["category_name"]
        score = det["score"]

        x, y, w, h = bbox
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=8)

        # Add label text
        label_text = f"{pred_label} ({score:.2f})"
        text_position = (x, y + h - font_size*0.01 - 2) if y > (font_size*0.01 + 2) else (x, y + h + 2)
        # text_position = (x, y + h - 20) if y >  20 else (x, y + h + 20)
        draw.text(text_position, label_text, fill="red", font=font)

    return img_copy

def visualize_bboxes(image_path, pred_bboxes, gt_bboxes, save_path):
    """
    Draw predicted bboxes in red, ground-truth bboxes in green on the image 
    and save to save_path.
    Bboxes should be [x, y, w, h] in image coordinates.
    """
    image = Image.open(image_path).convert("RGB")
    img_with_boxes = draw_bboxes_on_image(image, pred_bboxes, gt_bboxes)
    img_with_boxes.save(save_path)
    print(f"Saved visualization to {save_path}")

def create_img_with_bbox(original_image, bbox_xywh):
    """Draws a single red bounding box on an image."""
    img_with_bbox = original_image.copy()
    draw = ImageDraw.Draw(img_with_bbox)
    x, y, w, h = bbox_xywh
    bbox_xyxy = [x, y, x + w, y + h]
    draw.rectangle(bbox_xyxy, outline='red', width=3)
    return img_with_bbox


def get_masked_image_vqa_scores_with_instructions(qwen_model, qwen_processor, dataset_instructions_json, prompt_list, pil_images: list, batch_size: int = 8):
    """
    Scores a batch of images with bounding boxes based on a VQA prompt.
    This function is adapted from GridVQAscores_withSavedSAMProposal_webUI_RefCOCO_officialEval_saveInterimResults_gridWeightedBBox.py
    """
    if not pil_images: return np.array([])
    
    def getDatasetInstructions(dataset_instructions_json, class_name):
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

    # def getPrompt(prompt):
    #     # question = f"Is the main subject or object being referred to in this sentence: '{prompt}' located inside the red bounding box in the image? Please answer yes or no. Note: The object should be entirely inside the bounding box, with no part outside, and it must be the only object present inside - no other objects should appear within the box."
    #     question = f"Is the main subject or object being referred to as: '{prompt}' located inside the red bounding box in the image? Please answer Yes or No. Note: The object should be entirely inside the bounding box, with no part outside, and it must be the only object present inside - no other objects should appear within the box."
    #     return question


    def getPrompt(prompt, dataset_instructions_json):
        # # question = f"Is the main subject or object being referred to in this sentence: '{prompt}' located inside the red bounding box in the image? Please answer yes or no. Note: The object should be entirely inside the bounding box, with no part outside, and it must be the only object present inside - no other objects should appear within the box."
        # question = f"Is the main subject or object being referred to as: '{prompt}' located inside the red bounding box in the image? Please answer Yes or No. Note: The object should be entirely inside the bounding box, with no part outside, and it must be the only object present inside - no other objects should appear within the box."
        
        # question = f"""Is the main subject or object being referred to as: '{prompt}' located inside the red bounding box in the image? Please answer Yes or No. Note: The object should be entirely inside the bounding box, with no part outside, and it must be the only object present inside - no other objects should appear within the box.
        # The '{prompt}' class is described as follows in this context: {getDatasetInstructions(dataset_instructions_json, prompt)}"""
        
        # question = f"""
        #     Does the red bounding box in the image completely contain the main subject or object referred to as '{prompt}'? 
        #     Please answer only with "Yes" or "No".

        #     Requirements:
        #     - The '{prompt}' object must be **entirely inside** the red bounding box (no part should extend outside it).
        #     - The bounding box must contain **only this object** — no other objects should appear within it.

        #     Context: The '{prompt}' class is defined as follows:
        #     {getDatasetInstructions(dataset_instructions_json, prompt)}
        # """

        question = f"""
            Given the '{prompt}' class defined as follows: {getDatasetInstructions(dataset_instructions_json, prompt)}

            Is the main subject or object being referred to as: '{prompt}' located inside the red bounding box in the image? Please answer Yes or No. Note: The object should be entirely inside the bounding box, with no part outside, and it must be the only object present inside - no other objects should appear within the box.
        """

        return question

    all_final_scores = []
    # Process images in batches
    for i in range(0, len(pil_images), batch_size):
        batch_pil_images = pil_images[i:i + batch_size]
        batch_prompts = prompt_list[i:i + batch_size]
        
        # Create conversations for the batch
        # conversations = [[{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}] for img in batch_pil_images]
        # conversations = [[{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": getPrompt(prompt)}]}] for img, prompt in zip(batch_pil_images, batch_prompts)]
        conversations = [[{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": getPrompt(prompt, dataset_instructions_json)}]}] for img, prompt in zip(batch_pil_images, batch_prompts)]
        
        # Prepare inputs for the model
        text = qwen_processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(conversations)
        inputs = qwen_processor(text=text, images=image_inputs, padding=True, return_tensors="pt").to(qwen_model.device)
        
        # Generate outputs
        with torch.inference_mode():
            outputs = qwen_model.generate(**inputs, max_new_tokens=2, do_sample=False, output_scores=True, return_dict_in_generate=True)
        
        # Calculate 'Yes' probability
        scores = outputs.scores[0]
        probs = torch.nn.functional.softmax(scores, dim=-1)
        
        yes_token_id = qwen_processor.tokenizer.encode("Yes")[0]
        no_token_id = qwen_processor.tokenizer.encode("No")[0]
        
        yes_probs, no_probs = probs[:, yes_token_id], probs[:, no_token_id]
        # batch_scores = (yes_probs / (yes_probs + no_probs + 1e-9)).cpu().numpy()
        # batch_scores = (yes_probs / (yes_probs + no_probs + 1e-29)).cpu().numpy()
        # batch_scores = (yes_probs / (yes_probs + no_probs)).cpu().numpy()
        batch_scores = (yes_probs / (yes_probs + no_probs + 1e-18)).cpu().numpy()
        all_final_scores.extend(batch_scores.tolist())
    
    return np.array(all_final_scores)



def get_masked_image_vqa_scores(qwen_model, qwen_processor, prompt_list, pil_images: list, batch_size: int = 8):
    """
    Scores a batch of images with bounding boxes based on a VQA prompt.
    This function is adapted from GridVQAscores_withSavedSAMProposal_webUI_RefCOCO_officialEval_saveInterimResults_gridWeightedBBox.py
    """
    if not pil_images: return np.array([])
    
    def getPrompt(prompt):
        # question = f"Is the main subject or object being referred to in this sentence: '{prompt}' located inside the red bounding box in the image? Please answer yes or no. Note: The object should be entirely inside the bounding box, with no part outside, and it must be the only object present inside - no other objects should appear within the box."
        question = f"Is the main subject or object being referred to as: '{prompt}' located inside the red bounding box in the image? Please answer Yes or No. Note: The object should be entirely inside the bounding box, with no part outside, and it must be the only object present inside - no other objects should appear within the box."
        return question

    all_final_scores = []
    # Process images in batches
    for i in range(0, len(pil_images), batch_size):
        batch_pil_images = pil_images[i:i + batch_size]
        batch_prompts = prompt_list[i:i + batch_size]
        
        # Create conversations for the batch
        # conversations = [[{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}] for img in batch_pil_images]
        conversations = [[{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": getPrompt(prompt)}]}] for img, prompt in zip(batch_pil_images, batch_prompts)]
        
        # Prepare inputs for the model
        text = qwen_processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(conversations)
        inputs = qwen_processor(text=text, images=image_inputs, padding=True, return_tensors="pt").to(qwen_model.device)
        
        # Generate outputs
        with torch.inference_mode():
            outputs = qwen_model.generate(**inputs, max_new_tokens=2, do_sample=False, output_scores=True, return_dict_in_generate=True)
        
        # Calculate 'Yes' probability
        scores = outputs.scores[0]
        probs = torch.nn.functional.softmax(scores, dim=-1)
        
        yes_token_id = qwen_processor.tokenizer.encode("Yes")[0]
        no_token_id = qwen_processor.tokenizer.encode("No")[0]
        
        yes_probs, no_probs = probs[:, yes_token_id], probs[:, no_token_id]
        # batch_scores = (yes_probs / (yes_probs + no_probs + 1e-9)).cpu().numpy()
        # batch_scores = (yes_probs / (yes_probs + no_probs + 1e-29)).cpu().numpy()
        # batch_scores = (yes_probs / (yes_probs + no_probs)).cpu().numpy()
        batch_scores = (yes_probs / (yes_probs + no_probs + 1e-18)).cpu().numpy()
        all_final_scores.extend(batch_scores.tolist())
    
    return np.array(all_final_scores)


import string
def getClsIndex(predicted_tokens, num_cls):

    #Check for col - which is a character - but we muct also include cases like '[A' which is a single token
    cls_char = [f"{cls}" for cls in range(num_cls)]

    cls_index = np.ones(len(predicted_tokens))
    
    default_cls_idx = 1
    
    #Direct Match
    dir_cls_match = np.array([t in cls_char for t in predicted_tokens])
    
    found_cls = False

    if np.any(dir_cls_match):
        found_cls = True
        cls_index = dir_cls_match


    #Find any char or digit containing token
    if not found_cls:
        cls_index[np.where(np.array(predicted_tokens) == '<|im_end|>')] = 0

        # digit_index = [np.any([c.isdigit() and c not in string.punctuation for c in s]) for s in predicted_tokens]
        alpha_index = [np.any([c.isalpha() and c not in string.punctuation for c in s]) for s in predicted_tokens]

        # cls_index = np.logical_and(cls_index, digit_index)
        cls_index = np.logical_and(cls_index, alpha_index)

        if np.any(cls_index):
            found_cls = True


    #Handling default fallback
    if not found_cls:
        cls_index = np.ones(len(predicted_tokens))
        cls_index[default_cls_idx] = 1


    return cls_index


def get_masked_image_vqa_class_scores(qwen_model, qwen_processor, prompt_list, pil_images: list, class_name_list: list, batch_size: int = 8):
    """
    Scores a batch of images with bounding boxes based on a VQA prompt.
    This function is adapted from GridVQAscores_withSavedSAMProposal_webUI_RefCOCO_officialEval_saveInterimResults_gridWeightedBBox.py
    """
    if not pil_images: return np.array([])
    

    num_cls = len(class_name_list)

    def getPrompt(prompt):
        # # question = f"Is the main subject or object being referred to in this sentence: '{prompt}' located inside the red bounding box in the image? Please answer yes or no. Note: The object should be entirely inside the bounding box, with no part outside, and it must be the only object present inside - no other objects should appear within the box."
        # question = f"Is the main subject or object being referred to as: '{prompt}' located inside the red bounding box in the image? Please answer Yes or No. Note: The object should be entirely inside the bounding box, with no part outside, and it must be the only object present inside - no other objects should appear within the box."
        # question = f"Which grid in the image corresponds to the subject or object referred to in this sentence: '{prompt}'? Please only answer using a grid number (in Excel format: $colrow$) from the following: {str([[f"{chr(ord('A') + col)}{row}" for row in range(grid_size)] for col in range(grid_size)]).replace('[', '').replace(']', '')}" # * Best Prompts * - 1st best needs to be tested on entire dataset
        # question = f"Give the class name index the subject or object located inside the red bounding box in the image better relates to from the following: {str([f"[{i}]: {c}" for i,c in enumerate(class_name_list)])[1:-1].replace("'", '')}? Please only answer using the class name index number. Ex for class name: {class_name_list[0]}, output: [0]" #50,48 #* Best Prompts *
        # question = f"Identify which class the subject or object inside the red bounding box relates to the best from the following options: {str([f"[{i}]:{c}" for i,c in enumerate(class_name_list)])[1:-1].replace("'", '')}. Respond only with the class index number. For example, if the class is {class_name_list[0]}, output [0]." #47,46
        # question = f"Identify which class the subject or object inside the red bounding box belongs to from the following options: {str([f"[{i}]:{c}" for i,c in enumerate(class_name_list)])[1:-1].replace("'", '')}. Respond only with the class index number. For example, if the class is {class_name_list[0]}, output [0]." #50,48
        # question = f"Give the class name index the subject or object located inside the red bounding box in the image better relates to from the following: {str([f"[{i}]: {c}" for i,c in enumerate(class_name_list)])[1:-1].replace("'", '')}? Please only answer using the class name index number. Ex for class name: {class_name_list[0]}, output: [0]"
        
        # class_options = [f"[{chr(ord('A') + i)}]: {c}" for i, c in enumerate(class_name_list)]
        class_options = [f"${chr(ord('A') + i)}$: {c}" for i, c in enumerate(class_name_list)]
        class_options_str = ", ".join(class_options)
        # example_class_token = f"[{chr(ord('A'))}]"
        example_class_token = f"${chr(ord('A'))}$"
        
        # question = f"Identify which class the subject or object inside the red bounding box belongs to from the following options: {class_options_str}. Respond only with the class index letter. For example, if the class is {class_name_list[0]}, output {example_class_token}."
        question = f"Give the class name index the subject or object located inside the red bounding box in the image better relates to from the following: {class_options_str}? Please only answer using the class name index number. Ex for class name: {class_name_list[0]}, output: {example_class_token}."
        return question

    all_final_scores = []
    # Process images in batches
    for i in range(0, len(pil_images), batch_size):
        batch_pil_images = pil_images[i:i + batch_size]
        batch_prompts = prompt_list[i:i + batch_size]
        
        # Create conversations for the batch
        # conversations = [[{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}] for img in batch_pil_images]
        conversations = [[{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": getPrompt(prompt)}]}] for img, prompt in zip(batch_pil_images, batch_prompts)]
 
        # Prepare inputs for the model
        text = qwen_processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(conversations)
        inputs = qwen_processor(text=text, images=image_inputs, padding=True, return_tensors="pt").to(qwen_model.device)
        
        # Generate outputs
        with torch.inference_mode():
            # outputs = qwen_model.generate(**inputs, max_new_tokens=2, do_sample=False, output_scores=True, return_dict_in_generate=True)
            outputs = qwen_model.generate(**inputs, max_new_tokens=5, do_sample=False, output_scores=True, return_dict_in_generate=True)
        
        for b in range(len(batch_pil_images)):
            predicted_tokens = [qwen_processor.tokenizer.decode(torch.argmax(outputs.scores[i][b], dim=-1)) for i in range(len(outputs.scores))]
            print(f"[{b}] Predicted tokens: {predicted_tokens}")
    
            cls_index = getClsIndex(predicted_tokens, num_cls)
            print(f"[{b}] Found cls_index: {cls_index}")
            
            # We take the mean for all found tokens 
            cls_scores = torch.concat([outputs.scores[i][b].unsqueeze(0) for i in range(len(outputs.scores)) if cls_index[i] == 1], dim = 0).mean(dim = 0).unsqueeze(0)
            # cls_token_ids = [qwen_processor.tokenizer.encode(f"{cls}")[0] for cls in range(num_cls)]
            # cls_scores = torch.concat([outputs.scores[i][b].unsqueeze(0) for i, token_is_cls_idx in enumerate(cls_index) if token_is_cls_idx], dim=0).mean(dim=0).unsqueeze(0)
            # Get token IDs for 'A', 'B', 'C', etc.
            cls_token_ids = [qwen_processor.tokenizer.encode(f"{chr(ord('A') + i)}")[0] for i in range(num_cls)]
            print("[{b}] Cls-Tokens:" + str([f"Cls-{cls}: {t}" for cls, t in zip(range(num_cls), cls_token_ids)]))
        
            # Print the next predicted token 
            predicted_cls_token = qwen_processor.tokenizer.decode(torch.argmax(cls_scores, dim=-1))
            print("[{b}] Cls predicted token:", predicted_cls_token)
            
            cls_probs = torch.nn.functional.softmax(cls_scores, dim=-1)
            
            if len(cls_token_ids) != len(set(cls_token_ids)):
                print(f"\n\n******\n[{b}] Error! Cls Token ids aren't unique: {np.unique(cls_token_ids, return_counts = True)}\n******\n\n")

            cls_token_probs = [cls_probs[:, cls] for cls in cls_token_ids]
            print("[{b}] Cls-Tokens Probs:" + str([f"Cls-{cls}: {t}" for cls, t in zip(range(num_cls), cls_token_probs)]))

            #Normalize the col & row token probs - to avoid row/col domination
            # cls_token_probs = torch.softmax(torch.log(torch.tensor(cls_token_probs)), dim = 0)
            cls_token_probs = torch.tensor(cls_token_probs)
            cls_token_probs = cls_token_probs/cls_token_probs.sum()
            print("[{b}] Cls-Tokens Probs:" + str([f"Cls-{cls}: {t}" for cls, t in zip(range(num_cls), cls_token_probs)]))

            all_final_scores.append({'cls_name': class_name_list[cls_token_probs.argmax()], 'cls_prob':cls_token_probs.max()})
        
        # # Calculate 'Yes' probability
        # scores = outputs.scores[0]
        # probs = torch.nn.functional.softmax(scores, dim=-1)
        
        # yes_token_id = qwen_processor.tokenizer.encode("Yes")[0]
        # no_token_id = qwen_processor.tokenizer.encode("No")[0]
        
        # yes_probs, no_probs = probs[:, yes_token_id], probs[:, no_token_id]
        # batch_scores = (yes_probs / (yes_probs + no_probs + 1e-9)).cpu().numpy()
        # all_final_scores.extend(batch_scores.tolist())

    
    # return np.array(all_final_scores)
    return all_final_scores

def calculate_iou(boxA_xywh, boxB_xywh):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Boxes are expected in [x, y, w, h] format.
    """
    # Convert from [x, y, w, h] to [x1, y1, x2, y2]
    boxA = [boxA_xywh[0], boxA_xywh[1], boxA_xywh[0] + boxA_xywh[2], boxA_xywh[1] + boxA_xywh[3]]
    boxB = [boxB_xywh[0], boxB_xywh[1], boxB_xywh[0] + boxB_xywh[2], boxB_xywh[1] + boxB_xywh[3]]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0:
        return 0.0
        
    return interArea / unionArea

def apply_nms(detections, iou_threshold=0.5):
    """
    Applies Non-Maximum Suppression to a list of detections.
    Each detection is a dict with 'bbox' ([x,y,w,h]) and 'score'.
    """
    if not detections:
        return []

    # Sort detections by score in descending order
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)

    kept_detections = []
    while detections:
        # Keep the detection with the highest score
        best_det = detections.pop(0)
        kept_detections.append(best_det)

        # Remove detections that have a high IoU with the best one
        remaining_detections = []
        for det in detections:
            if calculate_iou(best_det['bbox'], det['bbox']) < iou_threshold:
                remaining_detections.append(det)
        detections = remaining_detections

    return kept_detections

# def run_inference_on_single_image(args, model, processor, image_path, dataset_instructions, class_name, no_instructions=False, few_shot_examples=None, output_dir="."):
# def run_inference_on_single_image(args, model, processor, image_path, dataset_instructions, class_name_list, 
def run_inference_on_single_image(args, model, processor, image_path, dataset_instructions_json, class_name_list, 
                                    no_instructions=False, few_shot_dict=None, output_dir=".", 
                                    eval_class_name=None):
    """
    Runs Qwen inference on a single image and parses the output.
    This is the common logic shared between Streamlit and CLI modes.
    """
    set_seed(args.seed)

    raw_output = ""
    parsed_bboxes = []
    all_few_shot_examples = []

    # for cat_id in cat_ids_for_image: #GRG: This iterates over each category in the image - which can artifically boost results as we ignore predictions for missing categories
    for class_name in class_name_list: #GRG: This iterates over all categories in the dataset - which can is the right thing to do here - but can penalize results as we expect the model to predict all categories in each image
        
        if eval_class_name is not None and class_name != eval_class_name:
            continue

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
        
        # cat_name_str = ds_cat_names[0] if len(ds_cat_names) > 0 else "unknown"

        if few_shot_dict:
            num_few_shot = 3
            # num_few_shot = min(3, len(few_shot_samples)) 
            few_shot_examples_for_cat = few_shot_dict.get(class_name, [])
            # few_shot_samples_i = random.sample(few_shot_samples, num_few_shot) if few_shot_examples else None
            few_shot_examples_for_cat_i = random.sample(few_shot_examples_for_cat, num_few_shot)
        else:
            # few_shot_samples_i = None
            few_shot_examples_for_cat_i = None
        all_few_shot_examples.extend(few_shot_examples_for_cat_i or [])
        

        set_seed(args.seed)
        
        # raw_output_i = run_qwen_inference(
        raw_output_i, input_width, input_height = run_qwen_inference(
            args,
            model, processor,
            image_path=image_path,
            dataset_instructions=dataset_instructions,
            class_name=class_name,
            # class_name_list=class_name_list,
            no_instructions=no_instructions,
            # few_shot_examples=
            few_shot_examples=few_shot_examples_for_cat_i
        )

        # parsed_bboxes = parse_qwen_output_to_detections(raw_output, class_name_list, output_dir=output_dir)
        parsed_bboxes_i = parse_qwen_output_to_detections(raw_output_i, [class_name], output_dir=output_dir)

        # Convert normalized coordinates to absolute coordinates - Ref-fix: https://github.com/QwenLM/Qwen3-VL/blob/2f25a646fb0f329647428eb8dacf19293de6f5d4/cookbooks/spatial_understanding.ipynb
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        for det in parsed_bboxes_i:
            bbox = det["bbox"]
            x, y, bw, bh = bbox
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            
            # Convert normalized coordinates to absolute coordinates
            abs_y1 = int(y1/input_height * height)
            abs_x1 = int(x1/input_width * width)
            abs_y2 = int(y2/input_height * height)
            abs_x2 = int(x2/input_width * width)    

            abs_w = abs_x2 - abs_x1
            abs_h = abs_y2 - abs_y1

            det["bbox"] = [abs_x1, abs_y1, abs_w, abs_h]

        # Accumulate results for all classes
        parsed_bboxes.extend(parsed_bboxes_i)
        raw_output += f"\n\n--- For class '{class_name}' ---\n{raw_output_i}"

    # --- VQA-based Re-scoring ---
    # Create copies for different evaluation paths
    detections_orig_no_nms = [det.copy() for det in parsed_bboxes]
    detections_vqa_no_nms = []

    if args.vqa_rescore and parsed_bboxes:
        original_image = Image.open(image_path).convert("RGB")
        
        # Create a list of images, each with one bounding box drawn
        vqa_images = [create_img_with_bbox(original_image, det["bbox"]) for det in parsed_bboxes]
        
        # Get VQA scores for all bboxes in a single batch call
        # We use the category name of the first detection as the prompt for the whole batch,
        # assuming all detections in this context are for the same class.
        # vqa_prompt = parsed_bboxes[0]["category_name"]
        vqa_prompts = [det["category_name"] for det in parsed_bboxes]
       
        # vqa_scores = get_masked_image_vqa_scores(
        #     model, processor, vqa_prompt, vqa_images, batch_size=args.vqa_batch_size
        # )
        if args.class_rescore:
            vqa_dict = get_masked_image_vqa_class_scores(
                model, processor, vqa_prompts, vqa_images, class_name_list, batch_size=args.vqa_batch_size
            )
        else:
            # vqa_scores = get_masked_image_vqa_scores(
            #     model, processor, vqa_prompts, vqa_images, batch_size=args.vqa_batch_size
            # )
            vqa_scores = get_masked_image_vqa_scores_with_instructions(
                model, processor, dataset_instructions_json, vqa_prompts, vqa_images, batch_size=args.vqa_batch_size
            )
        
        detections_vqa_no_nms = [det.copy() for det in detections_orig_no_nms]

        # Replace original scores with VQA scores
        # for i, det in enumerate(parsed_bboxes):
        for i, det in enumerate(detections_vqa_no_nms):
            det["model_score"] = det["score"]  # Keep original model score for reference

            if args.class_rescore:
                det["model_category_name"] = det["category_name"]  # Keep original model category name
                det["vqa_category_name"] = vqa_dict[i]['cls_name']
                det["category_name"] = vqa_dict[i]['cls_name']

                det["vqa_score"] = vqa_dict[i]['cls_prob'].item()
                det["score"] = vqa_dict[i]['cls_prob'].item()
            else:
                det["vqa_score"] = vqa_scores[i]
                det["score"] = vqa_scores[i]
    else:
        # If not VQA-rescoring, the VQA-based lists are the same as original
        detections_vqa_no_nms = [det.copy() for det in detections_orig_no_nms]
    
    # --- End of VQA-based Re-scoring ---

    # --- Non-Maximum Suppression ---
    detections_orig_with_nms = apply_nms(detections_orig_no_nms, iou_threshold=args.nms_threshold) if args.apply_nms else detections_orig_no_nms
    detections_vqa_with_nms = apply_nms(detections_vqa_no_nms, iou_threshold=args.nms_threshold) if args.apply_nms else detections_vqa_no_nms
    # --- End of NMS ---

    return raw_output, all_few_shot_examples, {
        "orig_no_nms": detections_orig_no_nms,
        "orig_with_nms": detections_orig_with_nms,
        "vqa_no_nms": detections_vqa_no_nms,
        "vqa_with_nms": detections_vqa_with_nms,
    }



# def run_qwen_inference(args, model, processor, image_path, dataset_instructions, class_name_list, no_instructions=False, few_shot_examples=None):
def run_qwen_inference(args, model, processor, image_path, dataset_instructions, class_name, no_instructions=False, few_shot_examples=None, image=None):
    """
    Given a model, processor, local image path, instructions (from README), 
    and the current image's filename, run Qwen2.5-VL and return the raw text output.
    """
    set_seed(args.seed)

    #image = Image.open(image_path).convert("RGB")
    if image is None:
        image = Image.open(image_path).convert("RGB")


    if no_instructions:

        prompt_text = (
        # *** Best so far ***

        # 40,48 [51,48 withVQA] f""" #** Best
        # f"""
        #     Follow the steps outlined in the pseudo code below on this image. 
        #     Do NOT return code or explanations — only output the final JSON list of bounding boxes.

        #     Pseudo code for reference:
        #     INPUT:
        #         - image
        #         - class_list = {class_name_list}  # list of classes to detect

        #     PROCESS:
        #         detections = []  # initialize empty list for detected objects

        #         for each class_name in class_list:
        #             # Step 1: Scan the image at multiple scales to detect both large and tiny objects
        #             multi_scale_regions = model.predict_regions_multiscale(image, class_name)

        #             # Step 2: For each candidate region, get bounding box and initial confidence score
        #             for region in multi_scale_regions:
        #                 bbox = region.get_bbox()  # [x_min, y_min, x_max, y_max]
        #                 bbox_confidence = region.get_confidence()  # confidence that bbox contains an object
        #                 class_cosine_similarity = model.get_class_similarity(region, class_name)  # cosine similarity of region to class_name

        #                 # Step 3: Combine both scores for final confidence
        #                 # - This ensures the score reflects both detection quality and label match
        #                 calibrated_score = bbox_confidence * 0.5 + class_cosine_similarity * 0.5  # weighted average (adjust weights if desired)


        #                 # Step 4: Include even small objects (tiny bounding boxes)
        #                 detections.append({{
        #                     "bbox_2d": bbox,
        #                     "label": class_name,
        #                     "score": calibrated_score
        #                 }})

        #         # Optional: sort detections by score descending
        #         detections.sort(key=lambda x: x['score'], reverse=True)

        #     OUTPUT:
        #         Return the 'detections' list in JSON format
        # """

        #2 35,53 [38,55 withVQA] f"For every class name in this list: {class_name_list}, find all objects (including living things in the image) that are related to it. For all detected objects score how semantically related it is to one of the class name on a scale from 0 to 1. Return the detected object's bounding box coordinates along with the score as a list of items like {{\"bbox_2d\":[x_min,y_min,x_max,y_max],\"label\":\"{class_name_list[0]}\",\"score\":*confidence_score 0-1*}}."  #**Best recall
        #2b 38,50 f"For every class label in this list: {class_name_list}, find all objects (including living things in the image) that are related to it. For all detected objects score how semantically related it is to one of the class label on a scale from 0 to 1. Return the detected object's bounding box coordinates along with the score as a list of items like {{\"bbox_2d\":[x_min,y_min,x_max,y_max],\"label\":\"{class_name_list[0]}\",\"score\":*confidence_score 0-1*}}."  #**1b Best**
        #1 38,46 f"For every object (including living things and text) in the image that are related to one of the class name in this list: {class_name_list}. For all detected objects score how semantically related it is to one of the class name on a scale from 0 to 1. Return the detected object's bounding box coordinates along with the score as a list of items like {{\"bbox_2d\":[x_min,y_min,x_max,y_max],\"label\":\"{class_name_list[0]}\",\"score\":*confidence_score 0-1*}}." #**1a Best**

        #1c 47,47 [41,47 withVQA] f"For every class label name in this list: {class_name_list}, find all objects (including living things in the image) that are related to it. For all detected objects score how semantically related it is to one of the class label name on a scale from 0 to 1. Return the detected object's bounding box coordinates along with the score as a list of items like {{\"bbox_2d\":[x_min,y_min,x_max,y_max],\"label\":\"{class_name_list[0]}\",\"score\":*confidence_score 0-1*}}. Note: 1) Avoid overlapping bounding boxes 2) No object with multiple class labels." #**Best**
        
        #Baseline-single-class: 47,52 f"Detect all of the subjects or objects that can be referred as '{class_name}' in the image and return their locations coordinates, as a list of items like {{\"bbox_2d\":[x_min,y_min,x_max,y_max],\"label\":\"{class_name}\",\"score\":*confidence_score 0-1*}}." #**2nd Best**
            
            # - class_list = {class_name_list}  # list of classes to detect
            
            f"""
            Follow the steps outlined in the pseudo code below on this image. 
            Do NOT return code or explanations — only output the final JSON list of bounding boxes.

            Pseudo code for reference:
            INPUT:
                - image
                - class_list = [{class_name}]  # list of classes to detect

            PROCESS:
                detections = []  # initialize empty list for detected objects

                for each class_name in class_list:
                    # Step 1: Scan the image at multiple scales to detect both large and tiny objects
                    multi_scale_regions = model.predict_regions_multiscale(image, class_name)

                    # Step 2: For each candidate region, get bounding box and initial confidence score
                    for region in multi_scale_regions:
                        bbox = region.get_bbox()  # [x_min, y_min, x_max, y_max]
                        bbox_confidence = region.get_confidence()  # confidence that bbox contains an object
                        class_cosine_similarity = model.get_class_similarity(region, class_name)  # cosine similarity of region to class_name

                        # Step 3: Combine both scores for final confidence
                        # - This ensures the score reflects both detection quality and label match
                        calibrated_score = bbox_confidence * 0.5 + class_cosine_similarity * 0.5  # weighted average (adjust weights if desired)


                        # Step 4: Include even small objects (tiny bounding boxes)
                        detections.append({{
                            "bbox_2d": bbox,
                            "label": class_name,
                            "score": calibrated_score
                        }})

                # Optional: sort detections by score descending
                detections.sort(key=lambda x: x['score'], reverse=True)

            OUTPUT:
                Return the 'detections' list in JSON format
        """
        )
    else:
        prompt_text = (

    # print(f"Using dataset instructions in the prompt: {dataset_instructions}")
    
    # prompt_text = (
    #     f"Detect all of the {class_name}s in the image and return their locations coordinates, use image dataset's annotator instruction for help."
    #     f"Here are the instructions:\n{dataset_instructions}\n\n"
    #     f"Return a list of items like {{\"bbox_2d\":[x1,y1,x2,y2],\"label\":\"{class_name}\",\"score\":*confidence_score 0-1*}}."
    # )
    # prompt_text = (
    #     # f"Detect all of the {class_name}s in the image and return their locations coordinates, use image dataset's annotator instruction for help."
    #     f"Detect all of the subjects or objects that can be referred as '{class_name}' in the image, use image dataset's annotator instruction for help. and return their locations coordinates, as a list of items like {{\"bbox_2d\":[x_min,y_min,x_max,y_max],\"label\":\"{class_name}\",\"score\":*confidence_score 0-1*}}." #**2nd Best**
    #     f"Here are the instructions:\n{dataset_instructions}\n\n"
    #     f"Return a list of items like {{\"bbox_2d\":[x1,y1,x2,y2],\"label\":\"{class_name}\",\"score\":*confidence_score 0-1*}}."
    # )

        # no-instruct = 40,48 [51,48 withVQA] f""" #** Best
        # with-instruct = [41,41 withVQA] f"""
        # with-instruct = [48,44 withVQA] f""" Use the following image dataset's annotator instruction for better understanding the class name definitions:\n{dataset_instructions}"
        # with-instruct = [46,43 withVQA] f""" Use the following dataset annotator instructions to understand class name definitions and apply them consistently:\n{dataset_instructions}"
        # with-instruct = [47,43 withVQA] f""" Use the following dataset's annotator instructions to better understand class name definitions and instructions for how to annotate the bounding boxes:\n{dataset_instructions}"
        # with-instruct = [46,43 withVQA] f""" Following the dataset's annotator instructions detect and draw the bounding boxes and for better understand class name definitions:\n{dataset_instructions}\n"
        # with-instruct = [48,44 withVQA] f""" Follow the dataset’s annotator instructions when detecting objects in this image. Use these instructions to correctly interpret and apply the class name definitions:\n{dataset_instructions}\n"
        # with-instruct = [48,44 withVQA] f""" Follow the steps outlined in the pseudo code below on this image by following class name definitions and dataset's annotator instructions outlined as follows:\n{dataset_instructions}\n"
        # with-instruct = [48,44 withVQA] f""" Follow the steps outlined in the pseudo code below on this image. Use the dataset’s annotator instructions and class name definitions provided here to guide detection and labeling:\n{dataset_instructions}\n"
        )
    prompt_text = (
        f"""
            Follow the steps outlined in the pseudo code below on this image for object detection. Use the dataset’s annotator instructions and class name definitions provided here to guide detection and labeling:\n{dataset_instructions}\n"

            Do NOT return code or explanations — only output the final JSON list of bounding boxes.

            
            Pseudo code for reference:
            INPUT:
                - image
                - class_list = [{class_name}]  # list of classes to detect

            PROCESS:
                detections = []  # initialize empty list for detected objects

                for each class_name in class_list:
                    # Step 1: Scan the image at multiple scales to detect both large and tiny objects
                    multi_scale_regions = model.predict_regions_multiscale(image, class_name)

                    # Step 2: For each candidate region, get bounding box and initial confidence score
                    for region in multi_scale_regions:
                        bbox = region.get_bbox()  # [x_min, y_min, x_max, y_max]
                        bbox_confidence = region.get_confidence()  # confidence that bbox contains an object
                        class_cosine_similarity = model.get_class_similarity(region, class_name)  # cosine similarity of region to class_name

                        # Step 3: Combine both scores for final confidence
                        # - This ensures the score reflects both detection quality and label match
                        calibrated_score = bbox_confidence * 0.5 + class_cosine_similarity * 0.5  # weighted average (adjust weights if desired)


                        # Step 4: Include even small objects (tiny bounding boxes)
                        detections.append({{
                            "bbox_2d": bbox,
                            "label": class_name,
                            "score": calibrated_score
                        }})

                # Optional: sort detections by score descending
                detections.sort(key=lambda x: x['score'], reverse=True)

            OUTPUT:
                Return the 'detections' list in JSON format

        """    
        #     Use the following image dataset's annotator instruction for better understanding the class name definitions:\n{dataset_instructions}"
        # """
    )

    if few_shot_examples and len(few_shot_examples) > 0:
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": (
        #                 f"Detect all of the {class_name}s in the image and return their locations coordinates, use image dataset's annotator instruction and 2 image examples for help."
        #                 f"Here are the instructions:\n{dataset_instructions}\n\n"
        #                 f"Here are 2 images with example detection(s) (in red boxes) of {class_name}s:"
        #             )},
        #             {"type": "image", "image": few_shot_examples[0]["viz_path"]},
        #             {"type": "image", "image": few_shot_examples[1]["viz_path"]},
        #             {"type": "text", "text": f"Return a list of items like {{\"bbox_2d\":[x1,y1,x2,y2],\"label\":\"{class_name}\",\"score\":*confidence_score 0-1*}}. Here is the image to detect {class_name}s in:"},
        #             {"type": "image", "image": image_path},
        #         ],
        #     }
        # ]
        messages = [
            {
                "role": "user",
                "content": [
                    # {"type": "text", "text": ( #0.036, 0.075 
                    #     # f"Detect all of the {class_name}s in the image and return their locations coordinates, use image dataset's annotator instruction and 2 image examples for help."
                    #     # f"""
                    #     #     Follow the steps outlined in the pseudo code below on this image for object detection. Use the dataset’s annotator instructions and class name definitions provided here to guide detection and labeling:\n{dataset_instructions}\n"
                    #     f"""
                    #         Follow the steps outlined in the pseudo code below on this query image for object detection. Use the dataset’s annotator instructions and class name definitions provided here to guide detection and labeling:\n{dataset_instructions}\n"
            
                    #         Do NOT return few_shot_examplescode or explanations — only output the final JSON list of bounding boxes.

                            
                    #         Pseudo code for reference:
                    #         INPUT:
                    #             - image
                    #             - class_list = {class_name_list}  # list of classes to detect

                    #         PROCESS:
                    #             detections = []  # initialize empty list for detected objects

                    #             for each class_name in class_list:
                    #                 # Step 1: Scan the image at multiple scales to detect both large and tiny objects
                    #                 multi_scale_regions = model.predict_regions_multiscale(image, class_name)

                    #                 # Step 2: For each candidate region, get bounding box and initial confidence score
                    #                 for region in multi_scale_regions:
                    #                     bbox = region.get_bbox()  # [x_min, y_min, x_max, y_max]
                    #                     bbox_confidence = region.get_confidence()  # confidence that bbox contains an object
                    #                     class_cosine_similarity = model.get_class_similarity(region, class_name)  # cosine similarity of region to class_name

                    #                     # Step 3: Combine both scores for final confidence
                    #                     # - This ensures the score reflects both detection quality and label match
                    #                     calibrated_score = bbox_confidence * 0.5 + class_cosine_similarity * 0.5  # weighted average (adjust weights if desired)


                    #                     # Step 4: Include even small objects (tiny bounding boxes)
                    #                     detections.append({{
                    #                         "bbox_2d": bbox,
                    #                         "label": class_name,
                    #                         "score": calibrated_score
                    #                     }})

                    #             # Optional: sort detections by score descending
                    #             detections.sort(key=lambda x: x['score'], reverse=True)

                    #         OUTPUT:
                    #             Return the 'detections' list in JSON format

                    #     """   
                    #     # f"Here are the instructions:\n{dataset_instructions}\n\n"
                    #     # f"Here are 2 images with example detection(s) (in red boxes) of {class_name}s:"
                    # )},
                    # {"type": "text", "text": f"Here is a reference image with example detection(s) in red bounding boxes of '{few_shot_examples[0]["category_name"]}' class:"},
                    # {"type": "image", "image": few_shot_examples[0]["viz_path"]},
                    # {"type": "text", "text": f"Here is a reference image with example detection(s) in red bounding boxes of '{few_shot_examples[1]["category_name"]}' class:"},
                    # {"type": "image", "image": few_shot_examples[1]["viz_path"]},
                    # {"type": "text", "text": f"Here is a reference image with example detection(s) in red bounding boxes of '{few_shot_examples[2]["category_name"]}' class:"},
                    # {"type": "image", "image": few_shot_examples[2]["viz_path"]},
                    # {"type": "text", "text": f"Here is the query image to be analyzed:"},
                    # {"type": "image", "image": image_path},
                    {"type": "text", "text": ( 
                        # f"Detect all of the {class_name}s in the image and return their locations coordinates, use image dataset's annotator instruction and 2 image examples for help."
                        # f"""
                        #     Follow the steps outlined in the pseudo code below on this image for object detection. Use the dataset’s annotator instructions and class name definitions provided here to guide detection and labeling:\n{dataset_instructions}\n"
                        f"""
                            Follow the steps outlined in the pseudo code below on the query image for object detection. Use the dataset’s annotator instructions and class name definitions provided here to guide detection and labeling:\n{dataset_instructions}\n"
            
                            Do NOT return few_shot_examplescode or explanations — only output the final JSON list of bounding boxes.

                            
                            Pseudo code for reference:
                            INPUT:
                                - image
                                - class_list = [{class_name}]  # list of classes to detect

                            PROCESS:
                                detections = []  # initialize empty list for detected objects

                                for each class_name in class_list:
                                    # Step 1: Scan the image at multiple scales to detect both large and tiny objects
                                    multi_scale_regions = model.predict_regions_multiscale(image, class_name)

                                    # Step 2: For each candidate region, get bounding box and initial confidence score
                                    for region in multi_scale_regions:
                                        bbox = region.get_bbox()  # [x_min, y_min, x_max, y_max]
                                        bbox_confidence = region.get_confidence()  # confidence that bbox contains an object
                                        class_cosine_similarity = model.get_class_similarity(region, class_name)  # cosine similarity of region to class_name

                                        # Step 3: Combine both scores for final confidence
                                        # - This ensures the score reflects both detection quality and label match
                                        calibrated_score = bbox_confidence * 0.5 + class_cosine_similarity * 0.5  # weighted average (adjust weights if desired)


                                        # Step 4: Include even small objects (tiny bounding boxes)
                                        detections.append({{
                                            "bbox_2d": bbox,
                                            "label": class_name,
                                            "score": calibrated_score
                                        }})

                                # Optional: sort detections by score descending
                                detections.sort(key=lambda x: x['score'], reverse=True)

                            OUTPUT:
                                Return the 'detections' list in JSON format

                        """   
                        # f"Here are the instructions:\n{dataset_instructions}\n\n"
                        # f"Here are 2 images with example detection(s) (in red boxes) of {class_name}s:"
                    )},
                    # {"type": "text", "text": f"Here are 3 images with example detection(s) in red bounding boxes of '{few_shot_examples[0]["category_name"]}' class, '{few_shot_examples[1]["category_name"]}' class, '{few_shot_examples[2]["category_name"]}' class respectively :"},
                    # {"type": "text", "text": f"Here are 3 images with example detection(s) (in red boxes) of {class_name}s:"},
                    # {"type": "text", "text": f"Here are 3 images with example detection(s) in red bounding boxes of '{class_name}' class:"},
                    {"type": "text", "text": f"Here are 3 images with example detection(s) in green bounding boxes of '{class_name}' class:"},
                    {"type": "image", "image": few_shot_examples[0]["viz_path"]},
                    {"type": "image", "image": few_shot_examples[1]["viz_path"]},
                    {"type": "image", "image": few_shot_examples[2]["viz_path"]},
                    # {"type": "text", "text": f"Now analyze the following query image by following the pseudo code above:"},
                    {"type": "text", "text": f"Now analyze the following query image by following the pseudo code above to detect {class_name}s in:"},
                    {"type": "image", "image": image_path},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image", "image": image_path},
                ],
            }
        ]
        # [Old 52,48 withVQA]
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": f"""
        #                                     Follow the steps outlined in the pseudo code below on this image. 
        #                                     Do NOT return code or explanations — only output the final JSON list of bounding boxes.

        #                                     Pseudo code for reference:
        #                                     INPUT:
        #                                         - image
        #                                     """
        #             },
        #             # {"type": "image", "image": image_path}, #[34,36 withVQA]
        #             {"type": "text", "text": f"""
        #                                         - class_list = {class_name_list}  # list of classes to detect

        #                                     PROCESS:
        #                                         detections = []  # initialize empty list for detected objects

        #                                         for each class_name in class_list:
        #                                             # Step 1: Scan the image at multiple scales to detect both large and tiny objects
        #                                             multi_scale_regions = model.predict_regions_multiscale(image, class_name)

        #                                             # Step 2: For each candidate region, get bounding box and initial confidence score
        #                                             for region in multi_scale_regions:
        #                                                 bbox = region.get_bbox()  # [x_min, y_min, x_max, y_max]
        #                                                 bbox_confidence = region.get_confidence()  # confidence that bbox contains an object
        #                                                 class_cosine_similarity = model.get_class_similarity(region, class_name)  # cosine similarity of region to class_name

        #                                                 # Step 3: Combine both scores for final confidence
        #                                                 # - This ensures the score reflects both detection quality and label match
        #                                                 calibrated_score = bbox_confidence * 0.5 + class_cosine_similarity * 0.5  # weighted average (adjust weights if desired)


        #                                                 # Step 4: Include even small objects (tiny bounding boxes)
        #                                                 detections.append({{
        #                                                     "bbox_2d": bbox,
        #                                                     "label": class_name,
        #                                                     "score": calibrated_score
        #                                                 }})

        #                                         # Optional: sort detections by score descending
        #                                         detections.sort(key=lambda x: x['score'], reverse=True)

        #                                     OUTPUT:
        #                                         Return the 'detections' list in JSON format
        #                                 """
        #             },
        #             # {"type": "image", "image": image_path}, #[40,42 withVQA]
        #             {"type": "text", "text": f"""
        #                                     Now, apply the following additional step to each bounding box in 'detections' to double its size before returning:
                     
        #                                     Pseudo code to pad bbox:

        #                                     INPUT:
        #                                         - detections  # list of detected objects with "bbox_2d"

        #                                     PROCESS:
        #                                         #Step 1: For each detection in detections:
        #                                         for detection in detections:        
        #                                             [x_min, y_min, x_max, y_max] = detection["bbox_2d"]
        #                                             width = x_max - x_min
        #                                             height = y_max - y_min
        #                                             center_x = x_min + width / 2
        #                                             center_y = y_min + height / 2
        #                                             # Step 2: Enlarge the bbox by a factor of 2
        #                                             new_width = width * 2
        #                                             new_height = height * 2
        #                                             # Step 3: Calculate new bbox coordinates
        #                                             new_x_min = max(0, center_x - new_width / 2)
        #                                             new_y_min = max(0, center_y - new_height / 2)
        #                                             new_x_max = min(image.width, center_x + new_width / 2)
        #                                             new_y_max = min(image.height, center_y + new_height / 2)
        #                                             # Step 4: Update detection with new bbox    
        #                                             detection["bbox_2d"] = [new_x_min, new_y_min, new_x_max, new_y_max]

        #                                     OUTPUT:
        #                                         Return detections strictly as a valid JSON list
        #                                     """
        #             },
        #             {"type": "image", "image": image_path}, #[40,42 withVQA]
        #         ],
        #     }
        # ]


    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # inputs['pixel_values'].shape
    # torch.Size([64680, 1176])
    
    # inputs['pixel_values'].shape
    # torch.Size([1380, 1176])

    with torch.no_grad():
        # generated_ids = model.generate(**inputs, max_new_tokens=512)
        # generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids = model.generate(**inputs, max_new_tokens=2048)

    # with torch.inference_mode():
    #     outputs = model.generate(
    #         **inputs,
    #         # max_new_tokens=1,
    #         # max_new_tokens=2,
    #         # max_new_tokens=5,
    #         max_new_tokens=10,
    #         do_sample=False,
    #         # # Set temp/top_p to neutral values for greedy search to suppress warnings
    #         # temperature=1.0,
    #         # top_p=1.0,
    #         output_scores=True,
    #         return_dict_in_generate=True
    #     )   
    
        
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    #Sample
    # ```json
    #     [
    #         {"bbox_2d": [1145, 563, 1427, 920], "label": "Adult", "score": 0.95}
    #     ]
    # ```

    #For scaling the bbox coordinates later
    input_height = inputs['image_grid_thw'][0][1]*14
    input_width = inputs['image_grid_thw'][0][2]*14

    # return output_text
    return output_text, input_width, input_height

# def parse_qwen_output_to_detections(output_text, output_dir="."):
def parse_qwen_output_to_detections(output_text, class_name_list, output_dir="."):
    """
    Robust parser for a Qwen output that looks like:
    ```json
    [
      {"bbox_2d": [x1, y1, x2, y2], "score": 0.95, "label": "some_label"},
      ...
    ]
    ```
    1) Removes code fences.
    2) Fixes missing colons in `bbox_2d [...]`.
    3) Parses the entire string as JSON (expecting a top-level list).
    4) If JSON parsing fails (e.g. due to truncation), it falls back to extracting individual JSON objects.
    5) Iterates each item; if one is malformed, it is skipped. Others are still accepted.

    Returns a list of detections, each:
      {"bbox": [x, y, w, h], "score": float, "category_name": str}
    
    It also logs any skipped or malformed items to `skipped_detections.log` in the output_dir.
    """
    skipped_log_path = os.path.join(output_dir, "skipped_detections.log")
    
    def log_skipped(reason, item, output_text):
        with open(skipped_log_path, "a") as f:
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reason": reason,
                "item": item,
                "original_output": output_text
            }
            f.write(json.dumps(log_entry) + "\n")

    # Remove code fences and fix bbox_2d formatting
    text_clean = re.sub(r'```(?:json)?\s*', '', output_text)
    text_clean = text_clean.replace('```', '')
    text_clean = re.sub(r'"bbox_2d\s*\[(.*?)\]', r'"bbox_2d":[\1]', text_clean)

    detections = []
    data = []
    try:
        data = json.loads(text_clean)
        if not isinstance(data, list):
            reason = "Top-level JSON is not a list"
            print(f"{reason}; skipping.")
            log_skipped(reason, text_clean, output_text)
            return detections
    except json.JSONDecodeError:
        reason = "Malformed JSON"
        print(f"{reason}, attempting to salvage valid detections.")
        print(text_clean)
        # Fallback: extract individual JSON objects using a regex.
        object_strings = re.findall(r'\{[^}]+\}', text_clean)
        for obj_str in object_strings:
            try:
                obj = json.loads(obj_str)
                data.append(obj)
            except json.JSONDecodeError:
                # print(f"Skipping malformed object: {obj_str}")
                try:
                    obj = json.loads(obj_str.replace('="', '":').replace(']"', ']').replace('"[', '['))
                    data.append(obj)
                except json.JSONDecodeError:
                    print(f"Skipping malformed object: {obj_str}")
                    reason = "Skipping malformed object during salvage"
                    print(f"{reason}: {obj_str}")
                    log_skipped(reason, obj_str, output_text)

    # Process each item in data (which should now be a list of dicts)
    for item in data:
        try:
            if not isinstance(item, dict):
                print(f"Skipping item (not a dict): {item}")
                reason = "Skipping item (not a dict)"
                print(f"{reason}: {item}")
                log_skipped(reason, item, output_text)
                continue

            bbox_2d = item.get("bbox_2d", [])
            if len(bbox_2d) != 4:
                print(f"Skipping invalid bbox_2d length: {bbox_2d}")
                reason = "Skipping invalid bbox_2d length"
                print(f"{reason}: {bbox_2d}")
                log_skipped(reason, item, output_text)
                continue

            # Convert bbox coordinates to floats (or ints as needed)
            try:
                x1, y1, x2, y2 = map(float, bbox_2d)
            except Exception as e:
                print(f"Skipping item due to conversion error: {bbox_2d}, error: {e}")
                reason = f"Skipping item due to conversion error: {e}"
                print(f"{reason}: {bbox_2d}")
                log_skipped(reason, item, output_text)
                continue

            if x2 < x1 or y2 < y1:
                print(f"Skipping reversed coords in bbox_2d: {bbox_2d}")
                reason = "Skipping reversed coords in bbox_2d"
                print(f"{reason}: {bbox_2d}")
                log_skipped(reason, item, output_text)
                continue
            w = x2 - x1
            h = y2 - y1
            if w == 0 or h == 0:
                print(f"Skipping zero dimension: {bbox_2d}")
                reason = "Skipping zero dimension"
                print(f"{reason}: {bbox_2d}")
                log_skipped(reason, item, output_text)
                continue

            label = item.get("label", "unknown")
            if label == "unknown":
                print(f"Skipping item (label is unknown): {item}")
                reason = "Skipping item (label is unknown)"
                print(f"{reason}: {item}")
                log_skipped(reason, item, output_text)
                continue

            #Multi-class: Check if label is in class_name_list (case-insensitive) - grg
            if label not in class_name_list:
                lb_check = [c for c in class_name_list if c.lower() == label.lower()]
                partial_lb_check = [c for c in class_name_list if c.lower() in label.lower()]
                if len(lb_check) > 0:
                    label = lb_check[0]  # Fix case mismatch
                elif len(partial_lb_check) > 0:
                    label = partial_lb_check[0]  # Fix partial match
                else:
                    # print(f"Skipping item (label is not in {class_name_list}): {item}")
                    # reason = f"Skipping item (label is not in {class_name_list})"
                    # print(f"{reason}: {item}")
                    # log_skipped(reason, item, output_text)
                    # continue

                    label = class_name_list[0]  # Assign default label and continue
                    print(f"Label is not in {class_name_list}): {item}, so assigning default label {class_name_list[0]} and continuing")
                    reason = f"Label is not in {class_name_list}): {item}, so assigning default label {class_name_list[0]} and continuing"
                    print(f"{reason}: {item}")
                    log_skipped(reason, item, output_text)
                    
            score = float(item.get("score", -1.0))
            if score == -1.0:
                print(f"Skipping item (score is -1.0): {item}")
                reason = "Skipping item (score is -1.0)"
                print(f"{reason}: {item}")
                log_skipped(reason, item, output_text)
                continue

            detections.append({
                "bbox": [x1, y1, w, h],
                "score": score,
                "category_name": label
            })

        except Exception as e:
            print(f"Skipping item due to error: {item}, error: {e}")
            reason = f"Skipping item due to unexpected error: {e}"
            print(f"{reason}: {item}")
            log_skipped(reason, item, output_text)

    return detections

def build_few_shot_dict(dataset_path, examples_per_class=2, coco_override=None):
    """
    For each category in the train set, pick up to 3 random images that contain it,
    visualize the ground truth bounding boxes, and store them in a dictionary:

      {
        "cat_name_1": [
           { "image_path": "...", "viz_path": "...", "bboxes": [...], ... },
           ...
        ],
        "cat_name_2": [...],
        ...
      }
    """

    train_dir = os.path.join(dataset_path, "train")
    train_ann = os.path.join(train_dir, "_annotations.coco.json")

    few_shot_dict = {}

    # If train annotations are missing, just return empty
    if not os.path.isfile(train_ann):
        print(f"No train annotations found for few-shot in {dataset_path}.")
        return few_shot_dict


    # coco_train = COCO(train_ann)
    if coco_override is not None:
        coco_train = coco_override
        print(f"Using coco_override for {dataset_path}.")
    else:
        coco_train = COCO(train_ann)


    cat_ids = coco_train.getCatIds()
    if not cat_ids:
        print(f"No categories in train for few-shot in {dataset_path}.")
        return few_shot_dict

    viz_dir = os.path.join("data_viz", "few_shot_examples")
    os.makedirs(viz_dir, exist_ok=True)

    # For each category, pick up to 3 images
    for cat_id in cat_ids:
        cat_name = coco_train.cats[cat_id]["name"]

        # All images that have this category
        img_ids = coco_train.getImgIds(catIds=[cat_id])
        if not img_ids:
            # No images for this category
            few_shot_dict[cat_name] = []
            continue

        # Randomly choose up to 3 images
        selected_img_ids = random.sample(img_ids, min(examples_per_class, len(img_ids)))
        examples_list = []

        for chosen_img_id in selected_img_ids:
            ann_ids = coco_train.getAnnIds(imgIds=[chosen_img_id], catIds=[cat_id])
            anns = coco_train.loadAnns(ann_ids)

            img_info_list = coco_train.loadImgs(chosen_img_id)
            if not img_info_list:
                continue

            img_info = img_info_list[0]
            image_path = os.path.join(train_dir, img_info["file_name"])
            if not os.path.isfile(image_path):
                continue

            # Visualize GT boxes for THIS category only, or for the entire image if you like:
            gt_bboxes = [ann["bbox"] for ann in anns]

            out_viz_path = os.path.join(
                viz_dir, f"few_shot_{cat_name}_{os.path.basename(img_info['file_name'])}"
            )

            visualize_bboxes(
                image_path=image_path,
                pred_bboxes=[],  # no predictions, only GT
                gt_bboxes=gt_bboxes,
                save_path=out_viz_path
            )

            example_dict = {
                "image_path": image_path,
                "viz_path": out_viz_path,
                "category_name": cat_name,
                "bboxes": gt_bboxes
            }
            examples_list.append(example_dict)

        # Store them in the dictionary keyed by cat_name
        few_shot_dict[cat_name] = examples_list

    return few_shot_dict

def evaluate_dataset(args, model, processor, dataset_path, no_instructions, few_shot_examples=False, run_name="", output_dir="results", 
                     eval_class_name=None, eval_cat_id=None, 
                     max_samples=None, 
                     dataset_instructions_override_json=None, coco_override=None):
    train_dir = os.path.join(dataset_path, "train")
    ann_path = os.path.join(train_dir, "_annotations.coco.json")
    # # readme_path = os.path.join(dataset_path, "README.roboflow.txt")
    # readme_path = os.path.join(dataset_path, "README.dataset.txt")
    readme_json_path = os.path.join("./data_instr/default", f"README.dataset_{os.path.basename(dataset_path)}.json")
    if not os.path.isfile(ann_path):
        print(f"No train annotations found in {train_dir}, skipping.")
        return None
    if few_shot_examples:
        # few_shot_dict = build_few_shot_dict(dataset_path, examples_per_class=2)
        few_shot_dict = build_few_shot_dict(dataset_path, examples_per_class=10)

        assert sum([len(v) for k,v in few_shot_dict.items()]) == len([f for f in os.listdir(train_dir) if f.split('.')[-1] in ['.png', 'jpeg', 'jpg']]), "Few-shot examples count does not match number of training images!"
    
        few_shot_samples = []
        [few_shot_samples.extend(v) for k,v in few_shot_dict.items() if len(v) > 0]
        print(f"Built few-shot examples dictionary with {len(few_shot_dict)} categories and {len(few_shot_samples)} total examples for {dataset_path}.")
    else:
        few_shot_dict = None

    # dataset_instructions = ""
    # if dataset_instructions_override is not None:
    #     dataset_instructions = dataset_instructions_override
    # else:
    #     if os.path.isfile(readme_path):
    #         with open(readme_path, "r", encoding="utf-8") as f:
    #             dataset_instructions = f.read()
    
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

    # if args.eval and all_predictions_exist:
    if args.eval and all_predictions_exist and not args.ipt_mode:
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
                print(f"Image {img_filename} has categories: {[coco_gt.cats[cat_id]["name"] for cat_id in cat_ids_for_image]}")
                # cat_names = [coco_gt.cats[cat_id]["name"] for cat_id in cat_ids_for_image]
                # print(f"Image {img_filename} has categories: {cat_names}")
                
                
                raw_output, few_shot_examples_used, all_detections = run_inference_on_single_image( #grg_changed
                    args,
                    model, processor,
                    image_path=image_path,
                    # dataset_instructions=dataset_instructions,
                    dataset_instructions_json = dataset_instructions_json,
                    # class_name=cat_name_str,
                    class_name_list=ds_cat_names, #GRG: Pass the entire list of category names
                    no_instructions=no_instructions,
                    # few_shot_examples=few_shot_examples_for_cat,
                    # few_shot_examples=few_shot_samples_i,
                    # few_shot_examples=few_shot_examples_for_cat_i,
                    few_shot_dict=few_shot_dict,
                    output_dir=output_dir,
                    eval_class_name=eval_class_name,
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
                
                # Yield results for Streamlit live updates
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
        eval_type: stats.tolist() for eval_type, stats in all_stats.items()
    }

    with open(eval_results_path, "w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, indent=2)

    print(f"Saved evaluation results to {eval_results_path}")

    return all_stats



def generate_initial_class_definition(args, model, processor, class_name, initial_instructions, few_shot_examples):
    """
    Uses the VLM to generate an initial textual definition of a class based on all GT examples.
    """

    set_seed(args.seed)

    if not few_shot_examples:
        return ""

    content = [
        {"type": "text", "text": 
        # f"""Based on the following example images showing '{class_name}' in green bounding boxes, describe the key visual characteristics of this class. 
        # Provide a concise, clear, and descriptive definition that could be used to instruct someone on how to identify these objects. 
        # Do not mention the bounding boxes or colors in your response.
        # """
        # f"""Describe the objects in the following example images shown in green bounding boxes, 
        # describe the collective key visual characteristics common and unique to these objects, such that it can be used for identification and detection. 
        # Provide a concise, clear, and descriptive definition that could be used to instruct someone on how to identify these objects. 
        # Do not mention the bounding boxes or colors in your response.
        # Sample description for reference: {initial_instructions}\n
        # """
        # f"""Analyze the example images and describe the objects highlighted in green bounding boxes. 
        #     Identify and summarize the key visual characteristics that are consistently observed across these objects, 
        #     noting any features that distinguish them from other object types. 
        #     Your goal is to produce a concise, clear, and descriptive definition that can be used to guide accurate identification and detection of this object class. 
        #     Avoid mentioning bounding boxes or colors in your response. 
        #     For reference, consider the following example description: \n{initial_instructions}\n
        # """
        # f"""Analyze the following images and describe the subjects or objects highlighted in green bounding boxes. 
        #     Identify and summarize the key visual characteristics that are consistently observed across these objects, 
        #     specifically calling out all features that distinguish them from other object in the scene. 
        #     Your goal is to produce a clear and descriptive definition that can be used to guide accurate identification and distinction of this object class from other objects in the scene. 
        #     Avoid mentioning bounding boxes or colors in your response. 
        #     For reference, consider the following example description: \n{initial_instructions}\n
        # """
        # f"""
        #     Analyze the following images and describe the subjects or objects highlighted in green bounding boxes. 
        #     Identify and summarize the key visual characteristics that are consistently present across these objects. 
        #     Focus on the distinctive visual features that set this object class apart from other elements in the scene. 

        #     Your goal is to produce a clear, detailed, and generalizable definition that can guide accurate recognition of this object class in future images and easily distinguishable from other objects in the scene. 
        #     Do not mention bounding boxes, colors, or any annotation details in your response.

        #     For reference, here is an example description:
        #     {initial_instructions}
        # """
        
        
        # f"""
        #     Analyze the following images and describe the subjects or objects highlighted in green bounding boxes. 
        #     Identify and summarize the key visual characteristics that are consistently observed across these objects. 
        #     Emphasize the distinctive features that clearly differentiate this object class from other elements in the scene.

        #     Your goal is to produce a clear, detailed, and generalizable definition that enables accurate recognition of this object class in future images and makes it easily distinguishable from other objects. 
        #     Do not mention bounding boxes, colors, or any annotation details in your response.

        #     For reference, here is an example description:
        #     {initial_instructions}
        # """

        # f"""
        #     Analyze the following images and describe the subjects or objects highlighted in green bounding boxes. 
        #     Identify and summarize the key visual characteristics that are consistently observed across these objects. 
        #     Emphasize the distinctive features that clearly differentiate this object class from other elements in the scene.

        #     Your goal is to produce a clear, detailed, and generalizable definition that enables accurate recognition of this object class in future images and makes it easily distinguishable from other objects. 
        #     Do not mention bounding boxes, colors, or any annotation details in your response.
        # """

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
        content.append({"type": "image", "image": example["image_path"]})

    messages = [{"role": "user", "content": content}]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text_input], images=image_inputs, padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    definition = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    # Clean up the definition
    definition = definition.replace(f"The visual characteristics of the '{class_name}' class are:", "").strip()
    definition = definition.replace(f"Definition of '{class_name}':", "").strip()
    
    print(f"Generated initial definition for '{class_name}': {definition}")
    return definition

def generate_class_definition(args, model, processor, class_name, current_instructions, few_shot_examples):
    """
    Uses the VLM to generate a textual definition of a class based on few-shot examples.
    """
    
    # set_seed(args.seed)
    
    if not few_shot_examples:
        return ""

    content = [
        # {"type": "text", "text": f"Based on the following example images showing '{class_name}', describe the key visual characteristics of this class. Provide a concise definition that could be used to instruct someone on how to identify these objects. Do not mention the bounding boxes."},
        # {"type": "text", "text": f"Based on the following example images showing '{class_name}' in green bounding boxes, describe the key visual characteristics of this class. Provide a concise definition that could be used to instruct someone on how to identify these objects. Do not mention the bounding boxes."},
        {"type": "text", "text": 
        #  f"""Improve the following class name definitions for better detecting '{class_name}' class shown in green bounding boxes.
        # 
        #  such that the objects in red bounding boxes are not included whereas the objects in the yellow bounding boxes are included.
        #  
        #  f"""Given the following class name definitions: {current_instructions}\n
        #  Based on the following example images, improve the class name definitions for better detecting '{class_name}' class shown in green bounding boxes. 
        #  The class name definition needs to be improved to describe the key visual characteristics of this class,
        #  such that the objects in the blue bounding boxes are included whereas the objects in the red bounding boxes are excluded.
        #  Provide a concise definition that could be used to instruct someone on how to identify these objects. 
        #  Return only the improved class name definitions containing all class definitions along with the improved '{class_name}' class definition.
        #  Do not mention the bounding boxes."""
        # f"""Refine and improve the following class name definitions used for object detection: {current_instructions}\n
        #  Based on the following example images, improve the class name definitions for better detecting '{class_name}' class shown in green bounding boxes. 
        #  The class name definition needs to be improved to describe the key visual characteristics of this class,
        #  such that the objects in the blue bounding boxes are included whereas the objects in the red bounding boxes are excluded.
        #  Provide a concise definition that could be used to instruct someone on how to identify these objects. 
        #  Return only the improved class name definitions containing all class definitions along with the improved '{class_name}' class definition.
        #  Do not mention the bounding boxes."""
        
        # f"""Refine and improve the following object class definitions used for object detection: {current_instructions}

        #     Using the provided example images, enhance the definition of the '{class_name}' class (highlighted with green bounding boxes). 
        #     Revise its description to clearly capture the key visual features that distinguish this class.

        #     Ensure the improved definition:
        #     - Includes objects similar to those shown with blue bounding boxes.
        #     - Excludes objects similar to those shown with red bounding boxes.

        #     Provide a concise, clear, and descriptive definition suitable for training or guiding object identification.
        #     Return only the complete set of class definitions, including the refined '{class_name}' definition.
        #     Do not refer to bounding boxes or colors in your response.
        # """

        # f"""Refine and improve the following object class definition of the '{class_name}' used in object detection: {current_instructions}

        #     Using the provided example images, enhance the definition of the '{class_name}' class (highlighted with green bounding boxes). 
        #     Revise its description to clearly capture the key visual features that distinguish this class.

        #     Ensure the improved definition:
        #     - Includes objects similar to those shown with blue bounding boxes.
        #     - Excludes objects similar to those shown with red bounding boxes.

        #     Provide a concise, clear, and descriptive definition suitable for training or guiding object identification.
        #     Return only the refined '{class_name}' class definition.
        #     Do not refer to bounding boxes or colors in your response.
        # """
        
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
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text_input], images=image_inputs, padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    definition = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    print(f"Generated definition for '{class_name}': {definition}")
    return definition



def generate_class_definition_withFP(args, model, processor, class_name, current_instructions, correct_image, FP_error_image):
    """
    Uses the VLM to generate a textual definition of a class based on few-shot examples.
    """
    
    # set_seed(args.seed)

    content = [
        # {"type": "text", "text": f"Based on the following example images showing '{class_name}', describe the key visual characteristics of this class. Provide a concise definition that could be used to instruct someone on how to identify these objects. Do not mention the bounding boxes."},
        # {"type": "text", "text": f"Based on the following example images showing '{class_name}' in green bounding boxes, describe the key visual characteristics of this class. Provide a concise definition that could be used to instruct someone on how to identify these objects. Do not mention the bounding boxes."},
        {"type": "text", "text": 
        
        # f"""Using the class definition for the '{class_name}' class, 
            
        #     the VLM model successfully identifies the '{class_name}' in the 'correct_image' shown in green bounding boxes. However, false positive detections are observed in the 'FP_error_image' shown in red bounding boxes, where the model mistakenly identifies an incorrect object.

        #     Compare and critique the ambiguity in the class definitions that may have contributed to this false positive, by analyzing the provided images.
        #     Specifically, identify which visual features in the 'FP_error_image' align with the current class definition. 
        #     Also, highlight the distinguishing features between the 'correct_image' and 'FP_error_image' in order to accurately represent the '{class_name}' class.
        #     Afterward, refine and enhance the class definition to improve its accuracy in identifying the '{class_name}' class by taking into consideration the critique and feedback from the analysis.

        #     Return the updated class definition in the following format: ```python\n{{'{class_name}': <updated definition>}}\n```.

        #     Do not mention bounding boxes, colors, or image annotations in your response.

        #     Class definition of the '{class_name}' class: {current_instructions}\n
        # """

        # f"""What sets apart the subject/object in the 'correct_image' shown in green bounding boxes from the subject/object in the 'FP_error_image' shown in red bounding boxes?
        #     List the key visual differences that distinguish the two.

        #     Now modify the class definition for the '{class_name}' class to better capture these distinguishing features,
        #     so that the model can more accurately identify the '{class_name}' class and avoid false positives like the one seen in the 'FP_error_image'.

        #     Return the updated class definition in the following format: ```python\n{{'{class_name}': <updated definition>}}\n```.

        #     Do not mention bounding boxes, colors, or image annotations in your response.

        #     Class definition of the '{class_name}' class: {current_instructions}\n
        # """

        # },
        # {"type": "text", "text": f"Here is the 'correct_image' showing the '{class_name}' class in green bounding boxes:"},
        # {"type": "image", "image": correct_image["image_path"]},
        # {"type": "text", "text": f"Here is the 'FP_error_image' showing the false positive for the '{class_name}' class in red bounding boxes:"},
        # {"type": "image", "image": FP_error_image["image_path"]}

        #  f"""What sets apart the subject/object shown in green bounding boxes from the subject/object shown in red bounding boxes?
        #     List the key visual differences that distinguish the two.

        #     Now modify the class definition for the '{class_name}' class to better capture these distinguishing features,
        #     so that the model can more accurately identify the '{class_name}' class and avoid false positives like the one seen in the 'FP_error_image'.

        #     Return the updated class definition in the following format: ```python\n{{'{class_name}': <updated definition>}}\n```.

        #     Do not mention bounding boxes, colors, or image annotations in your response.

        #     Class definition of the '{class_name}' class: {current_instructions}\n
        # """


        #  f"""What sets apart the subject/object shown in green bounding box from the subject/object shown in red bounding box?
        #     List the key visual differences that distinguish the two.

        #     Now modify the class definition for the '{class_name}' class of the object in green bounding box to better capture these distinguishing features,
        #     so that the model can more accurately identify the '{class_name}' class and avoid false positive detection like the one seen in the red bounding box.

        #     Return the updated class definition in descriptive text in the following format: ```python\n{{'{class_name}': <updated definition>}}\n```.

        #     Do not mention bounding boxes, or bounding box colors, or image annotations in your response.

        #     Class definition of the '{class_name}' class: {current_instructions}\n
        # """


        # f"""What sets apart the subject/object shown in green bounding box from the subject/object shown in red bounding box?
        #     List the key visual differences that distinguish the two.

        #     Come up with a class definition for the object in green bounding box inorder to identify and distinguish it from the object in red bounding box.
        #     Your definition should focus on the key visual features that differentiate the two objects.

        #     Now compare your class definition with the given class definition for the '{class_name}' class of the object in green bounding box. 
        #     Now given an updated class definition for the '{class_name}' class of the object in green bounding box resulting from your analysis of both the definitions,
        #     so that we can more accurately and easily identify the '{class_name}' class shown in green bounding box and avoid false positive detection like the one seen in the red bounding box.

        #     Return the updated class definition in descriptive text in the following format: ```python\n{{'{class_name}': <updated definition>}}\n```.

        #     Do not mention bounding boxes, or bounding box colors, or image annotations in your response.

        #     Class definition of the '{class_name}' class: {current_instructions}\n
        # """

        # f"""
        #     Analyze the image carefully and identify the key visual differences between the object shown in the green bounding box and the one shown in the red bounding box.

        #     1. Describe the distinguishing visual characteristics that set apart the object in the green bounding box from the object in the red bounding box.
        #     2. Based on these distinguishing traits, formulate a clear and descriptive class definition for the object in the green bounding box. This definition should focus on its unique visual and contextual features that help differentiate it from the object in the red bounding box.
        #     3. Compare your new class definition with the existing definition of the '{class_name}' class provided below:
        #     {current_instructions}
        #     4. Synthesize both definitions to produce an improved, more precise class definition for the '{class_name}' class. The updated definition should make it easier to accurately identify true instances of the '{class_name}' class while reducing false positives similar to the one seen in the red bounding box.

        #     Return the final updated class definition as descriptive text in the following format: ```python\n{{'{class_name}': <updated definition>}}\n```.

        # """


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
        {"type": "image", "image": correct_image["image_path"]},
        # {"type": "text", "text": f"Here is the 'FP_error_image' showing the false positive for the '{class_name}' class in red bounding boxes:"},
        {"type": "image", "image": FP_error_image["image_path"]}
    ]

    messages = [{"role": "user", "content": content}]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text_input], images=image_inputs, padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    definition = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    print(f"Generated definition for '{class_name}': {definition}")
    return definition



def generate_class_definition_withFN(args, model, processor, class_name, current_instructions, correct_image, FN_error_image):
    """
    Uses the VLM to generate a textual definition of a class based on few-shot examples.
    """
    
    # set_seed(args.seed)

    content = [
        # {"type": "text", "text": f"Based on the following example images showing '{class_name}', describe the key visual characteristics of this class. Provide a concise definition that could be used to instruct someone on how to identify these objects. Do not mention the bounding boxes."},
        # {"type": "text", "text": f"Based on the following example images showing '{class_name}' in green bounding boxes, describe the key visual characteristics of this class. Provide a concise definition that could be used to instruct someone on how to identify these objects. Do not mention the bounding boxes."},
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
        {"type": "image", "image": correct_image["image_path"]},
        # {"type": "text", "text": f"Here is the 'FP_error_image' showing the false positive for the '{class_name}' class in red bounding boxes:"},
        {"type": "image", "image": FN_error_image["image_path"]}
    ]

    messages = [{"role": "user", "content": content}]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text_input], images=image_inputs, padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    definition = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
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
                                    use_streamlit=False, num_samples=None):
    
    """
    Performs iterative prompt refinement.
    1. Generates class definitions from few-shot examples.
    2. Evaluates the dataset.
    3. Identifies worst-performing examples.
    4. Refines the prompt and repeats.
    """

    set_seed(args.seed)

    if use_streamlit:
        st.header("Iterative Prompt Refinement")
    
    # Initial setup
    # readme_path = os.path.join(dataset_path, "README.roboflow.txt")
    # readme_path = os.path.join(dataset_path, "README.dataset.txt")
    # initial_instructions = ""
    # if os.path.isfile(readme_path):
    #     with open(readme_path, "r", encoding="utf-8") as f:
    #         initial_instructions = f.read()

    # readme_json_path = os.path.join(dataset_path, "README.dataset_class_def.json")
    readme_json_path = os.path.join("./data_instr/default", f"README.dataset_{os.path.basename(dataset_path)}.json")
    class_instructions_json = {}
    if os.path.isfile(readme_json_path):
        with open(readme_json_path, "r", encoding="utf-8") as f:
            class_instructions_json = json.load(f)

    # few_shot_dict = build_few_shot_dict(dataset_path, examples_per_class=5)
    

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
        if use_streamlit:
            st.warning(f"Note: Using num_samples={num_samples} may limit the ground-truth examples available for generating initial class definitions.")
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

    # --- Step 0: Generate initial class definitions from all GT examples ---
    st.subheader("Step 0: Generating Initial Class Definitions from All GT Examples")
    
    
    refined_class_instructions_json = {}
    for cat_id in ds_cat_ids: #GRG: This iterates over all categories in the dataset - which can is the right thing to do here - but can penalize results as we expect the model to predict all categories in each image
        class_name = coco_gt.cats[cat_id]["name"]
    

        # --- Step 0: Generate class definition ---


        # --- Step 0: Generate class definition using positive samples only ---
        if use_streamlit:
            st.markdown(f"**Generating initial definition for '{class_name}'...**")
        

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

            img_with_boxes = draw_colored_bboxes_on_image(img, "green", gt_bboxes)

            img_viz_path = os.path.join(dataset_result_dir, f"few_shot_example_cls_{class_name}_initial_imId_{chosen_img_id}_file_{os.path.basename(img_info['file_name'])}.png")
            
            #Resize the image if too large
            max_dimension = (1920, 1080)  # Example max dimensions (width, height)
            img_with_boxes.thumbnail(max_dimension, Image.LANCZOS)
            print(f"file_{os.path.basename(img_info['file_name'])} resized img.size: {img_with_boxes.size}")
            
            #Save image
            img_with_boxes.save(img_viz_path)

            gt_examples_for_class.append({"image_path": img_viz_path})



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
        if use_streamlit:
            st.markdown(f"**Original definition for class '{class_name}':**")
            st.write(initial_instructions)

        #Save initial instructions as text file
        org_instructions_path = os.path.join(dataset_result_dir, f"{class_name}_original_definition.txt")
        with open(org_instructions_path, "w", encoding="utf-8") as f:
            f.write(initial_instructions)

        # To avoid overwhelming the model, let's use a random sample of up to 10 examples for generation
        # examples_to_use = random.sample(gt_examples_for_class, min(10, len(gt_examples_for_class)))
        examples_to_use = gt_examples_for_class
        if len(gt_examples_for_class) != 10:
            print(f"Warning! GT examples count does not match expected number - 10!")
        
        # Display a few of the examples being used
        cols = st.columns(min(10, len(examples_to_use)))
        for idx, ex in enumerate(examples_to_use):
            cols[idx].image(ex["image_path"], caption=f"GT Example for {class_name}", width=150)

        # Generate the definition
        initial_instructions = generate_initial_class_definition(args, model, processor, class_name, initial_instructions, examples_to_use)
        if use_streamlit:
            st.text_area(f"Generated Initial Definition for '{class_name}'", initial_instructions, height=100, key=f"init_def_{class_name}")

        # Save the generated initial instructions as text file
        init_def_path = os.path.join(dataset_result_dir, f"{class_name}_initial_with_only_gt_definition.txt")
        with open(init_def_path, "w", encoding="utf-8") as f:
            f.write(initial_instructions)
        


        # --- Step 0: Generate class definition using negative samples ---
        if use_streamlit:
            st.markdown(f"**Generating initial definition for '{class_name}' by comparing with other classes...**")


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
            
            other_img_with_boxes = draw_colored_bboxes_on_image(other_img, "red", other_gt_bboxes)

            other_img_viz_path = os.path.join(dataset_result_dir, f"few_shot_example_cls_{class_name}_initial_other_imId_{chosen_other_img_id}_file_{os.path.basename(other_img_info['file_name'])}.png")
            #Save image
            other_img_with_boxes.save(other_img_viz_path)

            fp_examples_for_class = {"image_path": other_img_viz_path}

            # positive_examples_for_class = random.choice(examples_to_use)
            if len(examples_to_use) > idx:
                positive_examples_for_class = examples_to_use[idx]
            else:
                positive_examples_for_class = random.choice(examples_to_use)


            if use_streamlit:
                st.markdown("**Examples from refinement:**")
                cols = st.columns(2)
                
                img_with_boxes = Image.open(positive_examples_for_class["image_path"]).convert("RGB")
                cols[0].image(img_with_boxes, caption=f"GT Example for {class_name}", width=300)
                cols[1].image(other_img_with_boxes, caption=f"Negative Example from {other_class_name}", width=300)


            # --- Refine prompt for next iteration ---
            with st.spinner(f"Idx {idx}: Refining class definition for '{class_name}' by comparing against {other_class_name}..."):

                #False-positive focused refinement
                fp_generated_definition_analysis = generate_class_definition_withFP(args, model, processor, class_name, initial_instructions, positive_examples_for_class, fp_examples_for_class)
            
                fp_generated_definition = extract_class_definition(fp_generated_definition_analysis, class_name)

                if fp_generated_definition:
                    initial_instructions = fp_generated_definition


                    # Save the generated initial instructions as text file
                    init_def_path = os.path.join(dataset_result_dir, f"{class_name}_initial_definition_with_FP_{other_class_name}.txt")
                    with open(init_def_path, "w", encoding="utf-8") as f:
                        f.write(initial_instructions)
                        
            if use_streamlit:
            
                st.text_area(f"Generated False-Positive based Class Definition Analysis by comparing against {other_class_name}", fp_generated_definition_analysis, height=400, key=f"{class_name}_gen_def_analysis_other_class_{other_class_name}_fp_{idx}")

                st.markdown("**Refined Instructions for Next Iteration:**")
                st.text_area(f"Instructions Iteration {idx+1}", initial_instructions, height=200, key=f"{class_name}_other_class_{other_class_name}_instr_{idx}")



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

        instruction_refinements = {}    
        for i in range(num_iterations):
            if use_streamlit:
                st.subheader(f"Iteration {i+1}/{num_iterations}")
               
            

            # stats_type = "vqa_with_nms"
            stats_type = "vqa_no_nms"
            # stats_type = "orig_with_nms"


            # --- Step 2: Evaluate with the current prompt ---
            with st.spinner(f"Iteration {i+1}: Evaluating dataset..."):
                run_name = f"ipt_iter_{i}"
                
                # UI placeholders for live visualization
                if use_streamlit:
                    st.markdown("---")
                    st.markdown(f"**Live Predictions for Iteration {i+1} / Class '{class_name}'**")
                    live_cols = st.columns(2)
                    live_image_placeholder = live_cols[0].empty()
                    live_info_placeholder = live_cols[1].empty()
                    progress_bar = st.progress(0)

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
                    coco_override=coco_gt if num_samples is not None else None # Pass the coco_gt with limited samples if applicable
                )

                #Restore seed state
                set_seed_from_state(seed_state)
                
                all_results_for_iter = []
                for j, result in enumerate(eval_generator):
                    all_results_for_iter.append(result)
                    
                    if use_streamlit:
                        # Update UI with live results
                        original_image = Image.open(result["image_path"]).convert("RGB")
                        # img_with_boxes = draw_bboxes_with_labels_on_image(original_image, result["parsed_detections"], result["gt_anns"], result["cat_dict"])
                        img_with_boxes = draw_bboxes_with_labels_on_image(original_image, result["all_detections"][stats_type], result["gt_anns"], result["cat_dict"])
                        live_image_placeholder.image(img_with_boxes, caption=f"Sample {j+1}/{num_eval_samples}: {os.path.basename(result['image_path'])}", width="stretch")
                        # live_info_placeholder.json(result["parsed_detections"])
                        live_info_placeholder.json(result["all_detections"][stats_type])
                        progress_bar.progress((j + 1) / num_eval_samples)

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
                if use_streamlit:
                    st.metric(label=f"mAP@.50-.95 ({stats_type}) for '{class_name}'", value=f"{ap50_95:.4f}")
                    st.metric(label=f"AR@1 ({stats_type}) for '{class_name}'", value=f"{ar1:.4f}")

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
                    if use_streamlit:
                        st.warning(f"mAP decreased from previous iteration ({prev_mAP:.4f} to {ap50_95:.4f}) for class '{class_name}'. Reverting to previous instructions.")
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
                    other_cls_iou = [calculate_iou(gt_box, pred_box) for gt_box in other_cls_gt_bboxes]

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

                    gt_iou_list = [calculate_iou(gt_box, pred_box) for gt_box in gt_bboxes]
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
            
            if use_streamlit:
                st.markdown("**Worst Performing Examples from this Iteration:**")
                cols = st.columns(len(worst_examples_map))
                
            few_shot_examples = {}
            for idx, (ex_type, ex) in enumerate(worst_examples_map.items()):
                if ex is None:
                    st.warning(f"No example found for {ex_type} in iteration {i+1} for class '{class_name}'.")
                    continue

                img = Image.open(ex['image_path']).convert("RGB")
                if ex_type == 'best_match':
                    # if not ex['gt_bbox']: continue
                    img_with_boxes = draw_colored_bboxes_on_image(img, "green", [ex['gt_bbox']])
                    caption = f"Best Match (score: {ex['best_score']:.2f})"
                
                elif ex_type == 'worst_fp':
                    # if not ex['pred_bbox']: continue
                    #TODO-GRG: We need to ensure that there aren't any pred boxes that match GT boxes here
                    #TODO-GRG: We also need to ensure that there aren't any pred boxes that are actually right but shown wrong as the gt label is not there due to few-shot

                    img_with_boxes = draw_colored_bboxes_on_image(img, "red", [ex['pred_bbox']])
                    caption = f"Worst FP (Error: {ex['fp_error']:.2f})"
                
                elif ex_type == 'worst_fn':
                    # if not ex['gt_bbox']: continue
                    img_with_boxes = draw_colored_bboxes_on_image(img, "blue", [ex['gt_bbox']])
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

                few_shot_examples[ex_type] = {"image_path": img_viz_path}

                if use_streamlit:    
                    cols[idx].image(img_with_boxes, caption=caption, width=300)




            # --- Step 4: Refine prompt for next iteration ---
            with st.spinner(f"Iteration {i+1}: Refining class definition for '{class_name}'..."):
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
            if use_streamlit:
    
                st.markdown("**Generated Class Definition Analysis:**")
                st.text_area(f"Generated False-Negative based Class Definition Analysis for Iteration {i+1}", fn_generated_definition_analysis, height=400, key=f"{class_name}_gen_def_analysis_fn_{i}")
                st.text_area(f"Generated False-Positive based Class Definition Analysis for Iteration {i+1}", fp_generated_definition_analysis, height=400, key=f"{class_name}_gen_def_analysis_fp_{i}")

                st.markdown("**Refined Instructions for Next Iteration:**")
                st.text_area(f"Instructions Iteration {i+1}", current_instructions, height=200, key=f"{class_name}_instr_{i}")


        #Display Initial Instructions
        if use_streamlit:
            st.subheader("Initial Instructions")
            st.text_area("Initial Instructions", initial_instructions, height=200)

        print(f"Initial instructions: \n{initial_instructions}")

        #Display Final Refined Instructions

        if use_streamlit:
            st.subheader("Final Refined Instructions")
            st.text_area("Final Refined Instructions", current_instructions, height=200)

        print(f"Final refined instructions: \n{current_instructions}")

        #Save final refined instructions
        refined_instructions_path = os.path.join(dataset_result_dir, f"refined_instructions_{dataset_name}_cls_{class_name}.txt")
        with open(refined_instructions_path, "w", encoding="utf-8") as f:
            f.write(current_instructions)
        print(f"Saved final refined instructions to {refined_instructions_path}")

        instruction_refinements["final_refined_instructions"] = current_instructions


        if use_streamlit:
            #Display and save best instructions
            st.subheader("Best Instructions Achieved During Iterations")
            st.text_area("Best Instructions", best_instructions, height=200)

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

    if use_streamlit:
        st.subheader("All Refined Class Instructions")
        st.json(refined_class_instructions_json)


    # Save all refined class instructions
    all_refined_instructions_path = os.path.join(result_dir, f"all_refined_class_instructions_{dataset_name}.json")
    with open(all_refined_instructions_path, "w", encoding="utf-8") as f:
        json.dump(refined_class_instructions_json, f, indent=2)
    print(f"Saved all refined class instructions to {all_refined_instructions_path}")


    # return current_instructions
    return refined_class_instructions_json


def format_coco_metrics(stats):
    """
    Formats the 12 COCO metrics into a readable string, similar to coco_eval.summarize().
    """
    metric_names = [
        ("Average Precision", "(AP)", "IoU=0.50:0.95", "all", 100),
        ("Average Precision", "(AP)", "IoU=0.50", "all", 100),
        ("Average Precision", "(AP)", "IoU=0.75", "all", 100),
        ("Average Precision", "(AP)", "IoU=0.50:0.95", "small", 100),
        ("Average Precision", "(AP)", "IoU=0.50:0.95", "medium", 100),
        ("Average Precision", "(AP)", "IoU=0.50:0.95", "large", 100),
        ("Average Recall", "(AR)", "IoU=0.50:0.95", "all", 1),
        ("Average Recall", "(AR)", "IoU=0.50:0.95", "all", 10),
        ("Average Recall", "(AR)", "IoU=0.50:0.95", "all", 100),
        ("Average Recall", "(AR)", "IoU=0.50:0.95", "small", 100),
        ("Average Recall", "(AR)", "IoU=0.50:0.95", "medium", 100),
        ("Average Recall", "(AR)", "IoU=0.50:0.95", "large", 100),
    ]
    
    formatted_string = ""
    for i, (title, type_str, iou, area, max_dets) in enumerate(metric_names):
        formatted_string += f"{title:<18} {type_str:<5} @[ IoU={iou:<9} | area={area:>6s} | maxDets={max_dets:>3d} ] = {stats[i]:0.3f}\n"
        
    return formatted_string

def run_streamlit_app(args):
    st.set_page_config(layout="wide", page_title="Qwen-VL Benchmark")
    st.title("Qwen 2.5-VL on RF100-VL Datasets")

    with st.sidebar:
        st.header("Evaluation Settings")
        # root_dir = st.text_input("Datasets Root Directory", "../rf100-vl/"
        # root_dir = st.text_input("Datasets Root Directory", "./datasets/rf100-vl/")
        root_dir = st.text_input("Datasets Root Directory", "./datasets/rf100-vl-fsod/")
        
        if not os.path.isdir(root_dir):
            st.error("Datasets root directory not found.")
            st.stop()

        dataset_options = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        # Filter dataset options to only those in rf20_datasets
        rf20_data_path = "./code/rf100vl/qwen-2.5-vl-rf-fsod-master/datasets_links_fixed.csv"
        rf20_datasets = pd.read_csv(rf20_data_path, header=None).values.flatten().tolist()[1:] #Skip header
        rf20_datasets = [i.split('/')[-2] for i in rf20_datasets]
        assert len(rf20_datasets) == 20, "Expected 20 datasets in rf20_datasets"

        filtered_dataset_paths = []
        for ds_path in dataset_options:
            for rf20 in rf20_datasets:
                if ds_path in rf20:
                    filtered_dataset_paths.append(ds_path)
                    break

        dataset_options = filtered_dataset_paths
        assert len(dataset_options) == 20, "Expected 20 datasets in dataset_paths after filtering"


        selected_dataset = st.selectbox("Select Dataset", dataset_options)
        
        num_samples = st.number_input("Number of samples to evaluate", 1, 1000, 1)
        
        
        st.header("Inference Settings")
        no_instructions = st.checkbox("No Instructions", value=args.no_instructions)
        few_shot = st.checkbox("Few-shot Examples", value=args.few_shot)
        # vqa_rescore = st.checkbox("VQA-based Re-scoring", value=args.vqa_rescore)
        vqa_rescore = st.checkbox("VQA-based Re-scoring", value=True)
        # apply_nms = st.checkbox("Apply NMS", value=args.apply_nms)
        apply_nms = st.checkbox("Apply NMS", value=True)
        nms_threshold = st.slider("NMS Threshold", 0.0, 1.0, 0.5, 0.05, help="IoU threshold for Non-Maximum Suppression. Higher values allow more overlap.")
        class_rescore = st.checkbox("VQA-based Class Re-scoring", value=args.class_rescore)
        show_labels = st.checkbox("Show Labels on Images", value=True)
        
        output_dir = st.text_input("Output Directory", value=args.output_dir)
        
        seed = st.number_input("Random Seed", 0, 99999, value=args.seed)

        # st.header("Prompt Tuning")
        # prompt_tuning_mode = st.checkbox("Enable Prompt Tuning")
        # custom_prompt = st.text_area("Custom Prompt Template", "Detect all of the subjects or objects that can be referred as '{class_name}' in the image and return their locations coordinates, as a list of items like {{\"bbox_2d\":[x_min,y_min,x_max,y_max],\"label\":\"{class_name}\",\"score\":*confidence_score 0-1*}}.", height=150)

        st.header("Iterative Prompt Tuning (IPT)")
        # ipt_mode = st.checkbox("Enable Iterative Prompt Tuning", value=args.ipt_mode)
        ipt_mode = st.checkbox("Enable Iterative Prompt Tuning", value=True)
        num_ipt_iterations = st.number_input("Number of IPT Iterations", 1, 20, 10)

        start_button = st.button("Start Evaluation")

    if start_button:
        st.header(f"Evaluating: {selected_dataset}")
        # Update args from UI
        args.no_instructions = no_instructions
        args.few_shot = few_shot
        args.vqa_rescore = vqa_rescore
        args.apply_nms = apply_nms
        args.nms_threshold = nms_threshold
        args.class_rescore = class_rescore
        args.show_labels = show_labels
        args.output_dir = output_dir
        args.seed = seed

        args.ipt_mode = ipt_mode

        # Load model
        with st.spinner("Loading Qwen model..."):
            # Use a key to ensure the cached resource is re-evaluated if device changes, though not user-configurable in UI here.
            model, processor = load_qwen_model_cached(qwen_device="cuda:0")

        if args.ipt_mode:
            # Run the iterative prompt refinement process
            dataset_path = os.path.join(root_dir, selected_dataset)
            dataset_instructions_override_json = iterative_prompt_refinement(
                args,
                model=model,
                processor=processor,
                dataset_path=dataset_path,
                num_iterations=num_ipt_iterations,
                use_streamlit=True,
                num_samples=num_samples
            )


        # Set seed for reproducibility
        set_seed(seed)

        # Prepare dataset paths
        dataset_path = os.path.join(root_dir, selected_dataset)

        # UI Placeholders
        progress_bar = st.progress(0)
        st.header("Live Results")
        
        col1, col2 = st.columns(2)
        image_placeholder = col1.empty()
        info_placeholder = col2.empty()
        few_shot_placeholder = st.empty()

        run_name = "_".join(filter(None, [args.no_instructions and "noinstr", args.few_shot and "fewshot", args.vqa_rescore and "vqa", args.class_rescore and "cls_rescore", args.apply_nms and f"nms{args.nms_threshold}"])) or "default"
        eval_generator = evaluate_dataset(
            args, model, processor, dataset_path, 
            no_instructions=args.no_instructions, 
            few_shot_examples=args.few_shot, 
            run_name=run_name, 
            output_dir=args.output_dir,
            max_samples=num_samples,
            dataset_instructions_override_json=dataset_instructions_override_json if args.ipt_mode else None
        )

        # Loop through the generator to get live results
        for i, result in enumerate(eval_generator):
            # Visualization
            original_image = Image.open(result["image_path"]).convert("RGB")
            if args.show_labels:
                img_with_boxes = draw_bboxes_with_labels_on_image(original_image, result["parsed_detections"], result["gt_anns"], result["cat_dict"])
            else:
                img_with_boxes = draw_bboxes_on_image(original_image, result["pred_bboxes"], result["gt_bboxes"])
            

            image_placeholder.image(img_with_boxes, caption=f"Sample {i+1}/{num_samples}: {os.path.basename(result['image_path'])}", width="stretch")

            info_text = f"**Qwen Output:**\n```\n{result['raw_output']}\n```\n\n"
            info_text += f"**Parsed Detections:**\n```json\n{json.dumps(result['parsed_detections'], indent=2)}\n```"
            
            info_placeholder.markdown(info_text)
            
            # Also show bboxes before NMS if NMS was applied
            if args.apply_nms:
                info_placeholder.markdown(f"**Parsed Detections (before NMS):**\n```json\n{json.dumps(result['parsed_detections_before_nms'], indent=2)}\n```")

            # Display few-shot examples if they were used
            if result.get("few_shot_examples_used"):
                with few_shot_placeholder.container():
                    st.subheader("Few-shot Examples Used for this Sample")
                    fs_cols = st.columns(len(result["few_shot_examples_used"]))
                    for idx, fs_example in enumerate(result["few_shot_examples_used"]):
                        fs_image = Image.open(fs_example["viz_path"])
                        fs_cols[idx].image(fs_image, caption=f"Example for: {fs_example['category_name']}")

            # progress_bar.progress((i + 1) / num_samples)
            progress_bar.progress((i + 1) / max(num_samples, i+1))

        # Final evaluation
        st.header("Final Evaluation Metrics")
        # The final stats are now calculated inside evaluate_dataset, so we just need to display them.
        # We can read the saved JSON file for this.
        eval_dir = os.path.join(args.output_dir, "evaluations", run_name)
        eval_results_path = os.path.join(eval_dir, f"evaluation_{selected_dataset}.json")
        if os.path.exists(eval_results_path):
            with open(eval_results_path, 'r') as f:
                all_stats_dict = json.load(f)

            st.subheader("COCO Metrics Comparison")

            # Create a DataFrame for easy comparison
            df_data = {
                "Metric": ["AP@.50:.95", "AP@.50"],
                "Original Score (no NMS)": [all_stats_dict["orig_no_nms"][0], all_stats_dict["orig_no_nms"][1]],
                "Original Score (w/ NMS)": [all_stats_dict["orig_with_nms"][0], all_stats_dict["orig_with_nms"][1]],
                "VQA Score (no NMS)": [all_stats_dict["vqa_no_nms"][0], all_stats_dict["vqa_no_nms"][1]],
                "VQA Score (w/ NMS)": [all_stats_dict["vqa_with_nms"][0], all_stats_dict["vqa_with_nms"][1]],
            }
            df = pd.DataFrame(df_data).set_index("Metric")
            st.dataframe(df.style.format("{:.4f}"))

            st.subheader("Detailed Metrics (VQA Score w/ NMS)")
            st.text(format_coco_metrics(all_stats_dict["vqa_with_nms"]))
        else:
            st.warning("Could not find final evaluation file. The full run might have been interrupted or failed.")

def run_single_dataset_evaluation(args):
    """
    Runs evaluation for a single dataset. This function is called by the dispatcher.
    """
    if not args.dataset_path or not os.path.isdir(args.dataset_path):
        print(f"Error: Invalid or missing --dataset_path: {args.dataset_path}")
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
    set_seed(args.seed)

    # os.makedirs(args.output_dir, exist_ok=True)

    # root_dir = "datasets"
    # root_dir = "../rf100-vl/"
    # root_dir = "./datasets/rf100-vl/"
    root_dir = "./datasets/rf100-vl-fsod/"
    if not os.path.isdir(root_dir):
        print(f"Root directory not found: {root_dir}")
        return

    model, processor = load_qwen_model(device_map_auto=args.device_map_auto)
    model.eval()

    print("=" * 60)
    print(f"Evaluating dataset: {args.dataset_path}")

    if args.ipt_mode:
        # Run the iterative prompt refinement process
        dataset_instructions_override_json = iterative_prompt_refinement(
            args,
            model=model,
            processor=processor,
            dataset_path=args.dataset_path,
            num_iterations=args.num_ipt_iterations,
            use_streamlit=False,
        )

    # ds_stats = list(evaluate_dataset(args, model, processor, args.dataset_path, no_instructions=args.no_instructions, few_shot_examples=args.few_shot, run_name=run_name, output_dir=args.output_dir))
    eval_generator = evaluate_dataset(args, model, processor, args.dataset_path, no_instructions=args.no_instructions, few_shot_examples=args.few_shot, run_name=run_name, output_dir=args.output_dir,
            dataset_instructions_override_json=dataset_instructions_override_json if args.ipt_mode else None)
    try:
        while True:
            next(eval_generator)
    except StopIteration as e:
        ds_stats = e.value

    if ds_stats and "vqa_with_nms" in ds_stats:
        # print(f"mAP (AP50-95) for {os.path.basename(args.dataset_path)}: {ds_stats[-1][0]:.4f}")
        print(f"mAP (AP50-95) for {os.path.basename(args.dataset_path)}: {ds_stats['vqa_with_nms'][0]:.4f}")
    else:
        print(f"Evaluation failed for {args.dataset_path}")

def get_available_gpus():
    """Detects available GPU IDs using torch.cuda."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))

def worker_process_datasets(gpu_id, task_queue, args):
    """
    Worker function that runs on a specific GPU.
    It loads the model once and then dynamically fetches datasets from the queue.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # This function is now running in a separate process, so we need to set the device
    # before loading the model.
    if gpu_id != -1:
        torch.cuda.set_device(gpu_id)
        qwen_device=f"cuda:{gpu_id}"
    else:
        qwen_device="cpu"

    print(f"[Worker on GPU {gpu_id}] Loading model...")
    # model, processor = load_qwen_model()
    model, processor = load_qwen_model(qwen_device, device_map_auto=args.device_map_auto)
    model.eval()
    print(f"[Worker on GPU {gpu_id}] Model loaded.")

    run_modes = []
    if args.no_instructions: run_modes.append("noinstr")
    if args.few_shot: run_modes.append("fewshot")
    if args.vqa_rescore: run_modes.append("vqa")
    if args.class_rescore: run_modes.append("cls_rescore")
    run_name = "_".join(run_modes) if run_modes else "default"

    while not task_queue.empty():
        try:
            dataset_path = task_queue.get(timeout=1)
            print(f"[Worker on GPU {gpu_id}] Processing dataset: {dataset_path}")


            if args.ipt_mode:
                # Run the iterative prompt refinement process
                dataset_instructions_override_json = iterative_prompt_refinement(
                    args,
                    model=model,
                    processor=processor,
                    dataset_path=dataset_path,
                    num_iterations=args.num_ipt_iterations,
                    use_streamlit=False,
                )

            # The generator needs to be consumed fully to run the evaluation
            eval_generator = evaluate_dataset(args, model, processor, dataset_path, no_instructions=args.no_instructions, few_shot_examples=args.few_shot, run_name=run_name, output_dir=args.output_dir,
                                               dataset_instructions_override_json=dataset_instructions_override_json if args.ipt_mode else None)
            for _ in eval_generator:
                pass # Consume the generator
        except Exception as e: # Catch queue.Empty or other exceptions
            print(f"[Worker on GPU {gpu_id}] Queue is empty or an error occurred: {e}. Exiting.")
            break

def run_cli_evaluation(args):

    #Create output directories
    os.makedirs(args.output_dir, exist_ok=True)

    # # --- Iterative Prompt Tuning (IPT) Mode ---
    # if args.ipt_mode:
    #     if not args.dataset_path:
    #         print("Error: --dataset_path must be specified when using --ipt_mode.")
    #         return

    #     print("--- Running in Iterative Prompt Tuning (IPT) Mode ---")
    #     root_dir = "./datasets/rf100-vl-fsod/"
    #     dataset_path = os.path.join(root_dir, args.dataset_path)
        
    #     model, processor = load_qwen_model(device_map_auto=args.device_map_auto)
        
    #     iterative_prompt_refinement(
    #         args, model, processor, dataset_path, 
    #         num_iterations=args.num_ipt_iterations, 
    #     )
    #     print("--- IPT Mode Finished ---")
    #     return

    # If a single dataset is specified, run it directly.
    if args.dataset_path:
        root_dir = "./datasets/rf100-vl-fsod/"
        args.dataset_path = os.path.join(root_dir, args.dataset_path)
        run_single_dataset_evaluation(args)
        return

    # --- Dispatcher Logic ---
    available_gpus = get_available_gpus()
    if args.gpu_ids:
        # Filter available GPUs by user-provided list
        gpu_ids = [g for g in available_gpus if g in args.gpu_ids]
        if not gpu_ids:
             print(f"Error: None of the specified GPUs {args.gpu_ids} are available. Available GPUs: {available_gpus}")
             return
    else:
        gpu_ids = available_gpus

    print(f"Using GPUs: {gpu_ids}")
    
    if not gpu_ids:
        print("No GPUs available. Running sequentially on CPU.")
        gpu_ids = [-1] # Use -1 to signify CPU


    # root_dir = "../rf100-vl/"
    # root_dir = "./datasets/rf100-vl/"
    root_dir = "./datasets/rf100-vl-fsod/"
    if not os.path.isdir(root_dir):
        print(f"Root directory not found: {root_dir}")
        return

    dataset_paths = []
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            dataset_paths.append(entry.path)

    all_stats = []
    dataset_results = []

    #Open csv file
    rf20_data_path = "./code/rf100vl/qwen-2.5-vl-rf-fsod-master/datasets_links_fixed.csv"
    rf20_datasets = pd.read_csv(rf20_data_path, header=None).values.flatten().tolist()[1:] #Skip header
    rf20_datasets = [i.split('/')[-2] for i in rf20_datasets]
    assert len(rf20_datasets) == 20, "Expected 20 datasets in rf20_datasets"

    filtered_dataset_paths = []
    for ds_path in dataset_paths:
        for rf20 in rf20_datasets:
            if ds_path.split('/')[-1] in rf20:
                filtered_dataset_paths.append(ds_path)
                break

    dataset_paths = filtered_dataset_paths
    assert len(dataset_paths) == 20, "Expected 20 datasets in dataset_paths after filtering"

    dataset_paths_org = dataset_paths.copy()

    #Check existing eval files and skip those datasets
    run_modes = []
    if args.no_instructions: run_modes.append("noinstr")
    if args.few_shot: run_modes.append("fewshot")
    if args.vqa_rescore: run_modes.append("vqa")
    if args.class_rescore: run_modes.append("cls_rescore")
    run_name = "_".join(run_modes) if run_modes else "default"

    dataset_paths_to_process = [] #grg_changed
    for dataset_path in dataset_paths: #grg_changed
        # Check for the final evaluation file to determine if the dataset has been fully processed.
        if os.path.isfile(os.path.join(
            args.output_dir, "evaluations", run_name, f"evaluation_{os.path.basename(dataset_path)}.json"
        )):
            print(f"Skipping {os.path.basename(dataset_path)} as predictions already exist.")
            continue
        dataset_paths_to_process.append(dataset_path)

    dataset_paths = dataset_paths_to_process
    print(f"Datasets to process ({len(dataset_paths)}): {[os.path.basename(p) for p in dataset_paths]}")
    
    # --- Process Pool for Dispatching ---
    from multiprocessing import Process
    import multiprocessing as mp

    num_gpus = len(gpu_ids)
    
    # Create a shared queue and add all datasets to it
    task_queue = mp.Queue()
    for ds_path in dataset_paths:
        task_queue.put(ds_path)

    processes = []
    # Use 'spawn' to avoid CUDA initialization issues in forked processes
    # This is crucial when using CUDA with multiprocessing
    mp.set_start_method('spawn', force=True)

    for i, gpu_id in enumerate(gpu_ids):
        p = Process(target=worker_process_datasets, args=(gpu_id, task_queue, args))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    print("\nAll dataset evaluations have been dispatched.")

    # --- Aggregation of results ---
    print("\nAggregating results...")
    run_modes = []
    if args.no_instructions:
        run_modes.append("noinstr")
    if args.few_shot:
        run_modes.append("fewshot")
    if args.vqa_rescore:
        run_modes.append("vqa")
    if args.class_rescore:
        run_modes.append("cls_rescore")
    if args.apply_nms:
        run_modes.append(f"nms{args.nms_threshold}")
    run_name = "_".join(run_modes) if run_modes else "default"

    eval_files = glob.glob(os.path.join(args.output_dir, "evaluations", run_name, "evaluation_*.json"))
    eval_types = ["orig_no_nms", "orig_with_nms", "vqa_no_nms", "vqa_with_nms"]
    all_stats_by_type = {eval_type: [] for eval_type in eval_types}
    dataset_results = []

    for eval_file in eval_files:
        with open(eval_file, 'r') as f:
            all_stats_dict = json.load(f)
        
        for eval_type, stats_list in all_stats_dict.items():
            all_stats_by_type[eval_type].append(stats_list)

        dataset_results.append({
            "dataset_name": os.path.basename(eval_file).replace("evaluation_", "").replace(".json", ""),
            "stats": all_stats_dict
        })

    if any(all_stats_by_type.values()):
        print("=" * 80)
        print(f"Average Metrics Across {len(eval_files)} Datasets")
        print("=" * 80)

        header = f"{'Metric':<12} | {'Orig (no NMS)':<15} | {'Orig (w/ NMS)':<15} | {'VQA (no NMS)':<15} | {'VQA (w/ NMS)':<15}"
        print(header)
        print("-" * len(header))

        mean_stats_all_types = {
            eval_type: np.mean(stats, axis=0) for eval_type, stats in all_stats_by_type.items() if stats
        }

        ap_50_95_line = f"{'AP@.50:.95':<12} | "
        ap_50_line =    f"{'AP@.50':<12} | "
        for eval_type in eval_types:
            stats = mean_stats_all_types.get(eval_type, [0.0] * 12)
            ap_50_95_line += f"{stats[0]:<15.4f} | "
            ap_50_line +=    f"{stats[1]:<15.4f} | "
        
        print(ap_50_95_line)
        print(ap_50_line)
        print("=" * 80)

        dataset_results.append({
            "dataset_name": "average",
            "stats": {eval_type: stats.tolist() for eval_type, stats in mean_stats_all_types.items()}
        })

        save_path = os.path.join(args.output_dir, f"all_datasets_metrics_{run_name}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(dataset_results, f, indent=2)
        print(f"\nSaved all dataset metrics + averages to {save_path}.")

    else:
        print("No evaluation files found to aggregate.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Run evaluation from CLI instead of Streamlit UI.")
    parser.add_argument("--no_instructions", action="store_true", help="Run inference with no instructions")
    parser.add_argument("--few_shot", action="store_true", help="Use 3 random few-shot examples from test set")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to a single dataset to evaluate. If not set, all datasets will be evaluated in parallel.")
    # parser.add_argument("--output_dir", type=str, default="results/rf100vl/rf20_tmp1f", help="Directory to save results and visuals.")
    parser.add_argument("--output_dir", type=str, default="results/rf100vl_IPT/rf20_IPT_singleclass_codePrompt_vqaScoreFixed_classRescoreFix_withNMS_v1_instr", help="Directory to save results and visuals.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None, help='List of GPU IDs to use for processing. e.g. --gpu_ids 0 1 4')
    # parser.add_argument('--vqa_batch_size', type=int, default=32, help='Batch size for VQA scoring of candidate masks.')
    parser.add_argument('--vqa_batch_size', type=int, default=8, help='Batch size for VQA scoring of candidate masks.')
    parser.add_argument("--vqa_rescore", action="store_true", help="Use VQA-based re-scoring of candidate masks")
    parser.add_argument("--apply_nms", action="store_true", help="Apply Non-Maximum Suppression to detections.")
    parser.add_argument("--nms_threshold", type=float, default=0.5, help="IoU threshold for Non-Maximum Suppression.")
    parser.add_argument("--class_rescore", action="store_true", help="Use VQA-based class re-scoring of candidate masks")
    parser.add_argument("--ipt_mode", action="store_true", help="Enable Iterative Prompt Tuning (requires --dataset_path).")
    parser.add_argument("--num_ipt_iterations", type=int, default=3, help="Number of iterations for IPT.")

    parser.add_argument("--device_map_auto", action="store_true", help="Use device_map='auto' for model loading. Overrides --qwen_device if set.")

    args = parser.parse_args()

    if args.eval:
        # The run_cli_evaluation function will handle setting the seed with the parsed args
        run_cli_evaluation(args)
    else:
        run_streamlit_app(args)