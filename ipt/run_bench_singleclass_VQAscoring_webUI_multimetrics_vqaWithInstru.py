'''
Run cmd: CUDA_VISIBLE_DEVICES=0 python code/rf100vl/qwen-2.5-vl-rf-fsod-master/run_bench_singleclass_VQAscoring_webUI.py --eval --dataset_path ../rf100-vl/ --vqa_rescore --no_instructions
Run cmd: CUDA_VISIBLE_DEVICES=0 python code/rf100vl/qwen-2.5-vl-rf-fsod-master/run_bench_singleclass_VQAscoring_webUI.py --eval --vqa_rescore --no_instructions --output_dir results/rf100vl/rf20_singleclass_codePrompt_vqaScore_v1 --gpu_ids 0 1 2 3 4 5 6 7
Run cmd: CUDA_VISIBLE_DEVICES=0 python code/rf100vl/qwen-2.5-vl-rf-fsod-master/run_bench_singleclass_VQAscoring_webUI.py --eval --vqa_rescore --class_rescore --no_instructions --output_dir results/rf100vl/rf20_singleclass_codePrompt_vqaScore_v1 --gpu_ids 0 1 2 3 4 5 6 7
Run cmd: CUDA_VISIBLE_DEVICES=0 python code/rf100vl/qwen-2.5-vl-rf-fsod-master/run_bench_singleclass_VQAscoring_webUI.py --eval --vqa_rescore --class_rescore --no_instructions --apply_nms --nms_threshold 0.5 --output_dir results/rf100vl/rf20_singleclass_codePrompt_vqaScore_v1 --gpu_ids 0 1 2 3 4 5 6 7
Run cmd: CUDA_VISIBLE_DEVICES=0 python code/rf100vl/qwen-2.5-vl-rf-fsod-master/run_bench_singleclass_VQAscoring_webUI.py --eval --vqa_rescore --class_rescore --no_instructions --apply_nms --nms_threshold 0.5 --dataset_path wb-prova --output_dir results/rf100vl_new/rf20_singleclass_codePrompt_vqaScore_classRescore_nms0.5_v1 --gpu_ids 0 1 2 3 4 5 6 7
Run cmd: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python code/rf100vl/qwen-2.5-vl-rf-fsod-master/run_bench_singleclass_VQAscoring_webUI.py --eval --vqa_rescore --few_shot --apply_nms --nms_threshold 0.5 --dataset_path wb-prova --output_dir results/rf100vl_fixedPadBug/rf20_singleclass_codePrompt_vqaScore_nms0.5_fewShot_v1 --device_map_auto
Run cmd: CUDA_VISIBLE_DEVICES=2 python code/rf100vl/qwen-2.5-vl-rf-fsod-master/run_bench_singleclass_VQAscoring_webUI_multimetrics.py --eval --vqa_rescore --apply_nms --nms_threshold 0.5 --dataset_path wb-prova --output_dir results/rf100vl_tmp2/rf20_singleclass_codePrompt_vqaScore_classRescore_nms0.5_v1 --vqa_batch_size 1
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
def load_qwen_model_cached(qwen_device = "cuda:0"): # For Streamlit
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
                                #   no_instructions=False, few_shot_examples=None, output_dir="."):
                                    no_instructions=False, few_shot_dict=None, output_dir="."):
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
def run_qwen_inference(args, model, processor, image_path, dataset_instructions, class_name, no_instructions=False, few_shot_examples=None):
    """
    Given a model, processor, local image path, instructions (from README), 
    and the current image's filename, run Qwen2.5-VL and return the raw text output.
    """
    set_seed(args.seed)

    #image = Image.open(image_path).convert("RGB")

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

            #  f"""
            #     Follow the steps outlined in the pseudo code below on this image for object detection. 
                
            #     Do NOT return code or explanations — only output the final JSON list of bounding boxes.

                
            #     Pseudo code for reference:
            #     INPUT:
            #         - image
            #         - class_name = "{class_name}"  # single class to detect

            #     PROCESS:
            #         detections = []  # initialize empty list for detected objects

            #         # Step 1: Scan the image at multiple scales to detect both large and tiny objects
            #         multi_scale_regions = model.predict_regions_multiscale(image, class_name)

            #         # Step 2: For each candidate region, get bounding box and initial confidence score
            #         for region in multi_scale_regions:
            #             bbox = region.get_bbox()  # [x_min, y_min, x_max, y_max]
            #             bbox_confidence = region.get_confidence()  # confidence that bbox contains an object
            #             class_cosine_similarity = model.get_class_similarity(region, class_name)  # semantic cosine similarity of region to class_name

            #             # Step 3: Combine both scores for final confidence
            #             # - This ensures the score reflects both detection quality and label match
            #             calibrated_score = bbox_confidence * 0.5 + class_cosine_similarity * 0.5  # weighted average (adjust weights if desired)

            #             # Step 4: Include even small objects (tiny bounding boxes)
            #             detections.append({{
            #                 "bbox_2d": bbox,
            #                 "label": class_name,
            #                 "score": calibrated_score
            #             }})

            #         # Step 5: Sort detections by confidence score (descending)
            #         detections.sort(key=lambda x: x['score'], reverse=True)

            #     OUTPUT:
            #         Return the 'detections' list in JSON format

            # """  

        )

    else:

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
        
                    # - class_list = {class_name_list}  # list of classes to detect
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
            # 
            # f"""
            #     Follow the steps outlined in the pseudo code below on this image for object detection. 
            #     Use the dataset’s annotator instructions and class name definitions provided here to guide detection and labeling:
            #     {dataset_instructions}
 
            #     Do NOT return code or explanations — only output the final JSON list of bounding boxes.

                
            #     Pseudo code for reference:
            #     INPUT:
            #         - image
            #         - class_name = "{class_name}"  # single class to detect

            #     PROCESS:
            #         detections = []  # initialize empty list for detected objects

            #         # Step 1: Scan the image at multiple scales to detect both large and tiny objects
            #         multi_scale_regions = model.predict_regions_multiscale(image, class_name)

            #         # Step 2: For each candidate region, get bounding box and initial confidence score
            #         for region in multi_scale_regions:
            #             bbox = region.get_bbox()  # [x_min, y_min, x_max, y_max]
            #             bbox_confidence = region.get_confidence()  # confidence that bbox contains an object
            #             class_cosine_similarity = model.get_class_similarity(region, class_name)  # semantic cosine similarity of region to class_name

            #             # Step 3: Combine both scores for final confidence
            #             # - This ensures the score reflects both detection quality and label match
            #             calibrated_score = bbox_confidence * 0.5 + class_cosine_similarity * 0.5  # weighted average (adjust weights if desired)

            #             # Step 4: Include even small objects (tiny bounding boxes)
            #             detections.append({{
            #                 "bbox_2d": bbox,
            #                 "label": class_name,
            #                 "score": calibrated_score
            #             }})

            #         # Step 5: Sort detections by confidence score (descending)
            #         detections.sort(key=lambda x: x['score'], reverse=True)

            #     OUTPUT:
            #         Return the 'detections' list in JSON format

            # """    
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

def build_few_shot_dict(dataset_path, examples_per_class=2):
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

def evaluate_dataset(args, model, processor, dataset_path, no_instructions, few_shot_examples=False, run_name="", output_dir="results", max_samples=None):
    test_dir = os.path.join(dataset_path, "test")
    ann_path = os.path.join(test_dir, "_annotations.coco.json")
    # readme_path = os.path.join(dataset_path, "README.dataset.txt")
    # readme_json_path = os.path.join("./data_instr/default", f"README.dataset_{os.path.basename(dataset_path)}.json")
    readme_json_path = os.path.join(f"{args.data_instr_path}_{os.path.basename(dataset_path)}.json")
    if not os.path.isfile(ann_path):
        print(f"No test annotations found in {test_dir}, skipping.")
        return None
    if few_shot_examples:
        # few_shot_dict = build_few_shot_dict(dataset_path, examples_per_class=2)
        few_shot_dict = build_few_shot_dict(dataset_path, examples_per_class=10)

        assert sum([len(v) for k,v in few_shot_dict.items()]) == len([f for f in os.listdir(test_dir.replace('test', 'train')) if f.split('.')[-1] in ['.png', 'jpeg', 'jpg']]), "Few-shot examples count does not match number of training images!"
    
        few_shot_samples = []
        [few_shot_samples.extend(v) for k,v in few_shot_dict.items() if len(v) > 0]
        print(f"Built few-shot examples dictionary with {len(few_shot_dict)} categories and {len(few_shot_samples)} total examples for {dataset_path}.")
    else:
        few_shot_dict = None

    # dataset_instructions = ""
    # if os.path.isfile(readme_path):
    #     with open(readme_path, "r", encoding="utf-8") as f:
    #         dataset_instructions = f.read()
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
    if args.eval and any(os.path.isfile(p) for p in prediction_cache_paths.values()):
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
                    output_dir=output_dir
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
                if total_count % 10 == 0:
                    print(f"\nSaving intermediate results at image {total_count + 1}/{len(images_to_process)}...")
                    for eval_type, detections in detections_all_by_type.items():
                        with open(prediction_cache_paths[eval_type], "w", encoding="utf-8") as f:
                            json.dump(detections, f)
                
                
                # Yield results for Streamlit live updates
                yield {
                    "img_id": img_id,
                    "image_path": image_path,
                    "gt_bboxes": [ann["bbox"] for ann in anns],
                    "pred_bboxes": [det["bbox"] for det in all_detections["vqa_with_nms"]],
                    "raw_output": raw_output,
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
        eval_type: stats.tolist() for eval_type, stats in all_stats.items()
    }

    with open(eval_results_path, "w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, indent=2)

    print(f"Saved evaluation results to {eval_results_path}")

    return all_stats

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
    st.set_page_config(layout="wide")
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
        
        num_samples = st.number_input("Number of samples to evaluate", 1, 1000, 20)
        
        
        st.header("Inference Settings")
        no_instructions = st.checkbox("No Instructions", value=args.no_instructions)
        few_shot = st.checkbox("Few-shot Examples", value=args.few_shot)
        vqa_rescore = st.checkbox("VQA-based Re-scoring", value=args.vqa_rescore)
        apply_nms = st.checkbox("Apply NMS", value=args.apply_nms)
        nms_threshold = st.slider("NMS Threshold", 0.0, 1.0, 0.5, 0.05, help="IoU threshold for Non-Maximum Suppression. Higher values allow more overlap.")
        class_rescore = st.checkbox("VQA-based Class Re-scoring", value=args.class_rescore)
        show_labels = st.checkbox("Show Labels on Images", value=True)
        
        output_dir = st.text_input("Output Directory", value=args.output_dir)
        
        seed = st.number_input("Random Seed", 0, 99999, value=args.seed)

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

        # Load model
        with st.spinner("Loading Qwen model..."):
            model, processor = load_qwen_model_cached()

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
            max_samples=num_samples
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

    # ds_stats = list(evaluate_dataset(args, model, processor, args.dataset_path, no_instructions=args.no_instructions, few_shot_examples=args.few_shot, run_name=run_name, output_dir=args.output_dir))
    eval_generator = evaluate_dataset(args, model, processor, args.dataset_path, no_instructions=args.no_instructions, few_shot_examples=args.few_shot, run_name=run_name, output_dir=args.output_dir)

    # try:
    #     while True:
    #         next(eval_generator)
    # except StopIteration as e:
    #     ds_stats = e.value

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
            dataset_basename = os.path.basename(args.dataset_path.rstrip('/'))
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
        print(f"[orig_no_nms] mAP (AP50-95) for {os.path.basename(args.dataset_path)}: {ds_stats['orig_no_nms'][0]:.4f}")
        print(f"[orig_with_nms] mAP (AP50-95) for {os.path.basename(args.dataset_path)}: {ds_stats['orig_with_nms'][0]:.4f}")
        print(f"[vqa_no_nms] mAP (AP50-95) for {os.path.basename(args.dataset_path)}: {ds_stats['vqa_no_nms'][0]:.4f}")
        print(f"[vqa_with_nms] mAP (AP50-95) for {os.path.basename(args.dataset_path)}: {ds_stats['vqa_with_nms'][0]:.4f}")

        #AR@1
        print(f"[orig_no_nms] AR@1 for {os.path.basename(args.dataset_path)}: {ds_stats['orig_no_nms'][6]:.4f}")
        print(f"[orig_with_nms] AR@1 for {os.path.basename(args.dataset_path)}: {ds_stats['orig_with_nms'][6]:.4f}")
        print(f"[vqa_no_nms] AR@1 for {os.path.basename(args.dataset_path)}: {ds_stats['vqa_no_nms'][6]:.4f}")
        print(f"[vqa_with_nms] AR@1 for {os.path.basename(args.dataset_path)}: {ds_stats['vqa_with_nms'][6]:.4f}")

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
            # The generator needs to be consumed fully to run the evaluation
            eval_generator = evaluate_dataset(args, model, processor, dataset_path, no_instructions=args.no_instructions, few_shot_examples=args.few_shot, run_name=run_name, output_dir=args.output_dir)
            for _ in eval_generator:
                pass # Consume the generator
        except Exception as e: # Catch queue.Empty or other exceptions
            print(f"[Worker on GPU {gpu_id}] Queue is empty or an error occurred: {e}. Exiting.")
            break

def run_cli_evaluation(args):

    #Create output directories
    os.makedirs(args.output_dir, exist_ok=True)

    # If a single dataset is specified, run it directly.
    if args.dataset_path:
        # root_dir = "./datasets/rf100-vl/"
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

    dataset_paths_to_process = []
    for dataset_path in dataset_paths:

        # if os.path.isfile(predictions_cache_path):
        if os.path.isfile(os.path.join(
            args.output_dir, "predictions", run_name, f"predictions_{os.path.basename(dataset_path)}.json"
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
    
    for eval_file in eval_files:
        with open(eval_file, 'r') as f:
            stats_dict = json.load(f)
        stats_list = [stats_dict.get(k, 0.0) for k in ["AP_50_95", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large", "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"]]
        all_stats.append(stats_list)
        dataset_results.append({
            "dataset_name": os.path.basename(eval_file).replace("evaluation_", "").replace(".json", ""),
            "stats": stats_list
        })

    if len(all_stats) > 0:
        mean_stats = np.mean(all_stats, axis=0)

        print("=" * 60)
        print(f"Average across {len(all_stats)} datasets:\n")

        print(f"  AP (IoU=0.50:0.95):  {mean_stats[0]:.4f}")
        print(f"  AP (IoU=0.50):       {mean_stats[1]:.4f}")
        print(f"  AP (IoU=0.75):       {mean_stats[2]:.4f}")
        print(f"  AP (small):          {mean_stats[3]:.4f}")
        print(f"  AP (medium):         {mean_stats[4]:.4f}")
        print(f"  AP (large):          {mean_stats[5]:.4f}")
        print(f"  AR@1:                {mean_stats[6]:.4f}")
        print(f"  AR@10:               {mean_stats[7]:.4f}")
        print(f"  AR@100:              {mean_stats[8]:.4f}")
        print(f"  AR (small):          {mean_stats[9]:.4f}")
        print(f"  AR (medium):         {mean_stats[10]:.4f}")
        print(f"  AR (large):          {mean_stats[11]:.4f}")

        dataset_results.append({
            "dataset_name": "average",
            "stats": mean_stats.tolist()
        })

        save_path = os.path.join(args.output_dir, "all_datasets_metrics.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(dataset_results, f, indent=2)
        print(f"\nSaved all dataset metrics + averages to {save_path}.")

    else:
        print("No valid datasets found or no predictions made.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Run evaluation from CLI instead of Streamlit UI.")
    parser.add_argument("--no_instructions", action="store_true", help="Run inference with no instructions")
    parser.add_argument("--few_shot", action="store_true", help="Use 3 random few-shot examples from test set")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to a single dataset to evaluate. If not set, all datasets will be evaluated in parallel.")
    # parser.add_argument("--output_dir", type=str, default="results/rf100vl/rf20_tmp1f", help="Directory to save results and visuals.")
    parser.add_argument("--output_dir", type=str, default="results/rf100vl-zeroshot/rf20_singleclass_codePrompt_vqaScore_withNMS_v1_instr_withPerClassInstrc", help="Directory to save results and visuals.")
    parser.add_argument("--data_instr_path", type=str, default="./data_instr/default/README.dataset", help="Directory to save results and visuals.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None, help='List of GPU IDs to use for processing. e.g. --gpu_ids 0 1 4')
    parser.add_argument('--vqa_batch_size', type=int, default=32, help='Batch size for VQA scoring of candidate masks.')
    parser.add_argument("--vqa_rescore", action="store_true", help="Use VQA-based re-scoring of candidate masks")
    parser.add_argument("--apply_nms", action="store_true", help="Apply Non-Maximum Suppression to detections.")
    parser.add_argument("--nms_threshold", type=float, default=0.5, help="IoU threshold for Non-Maximum Suppression.")
    parser.add_argument("--class_rescore", action="store_true", help="Use VQA-based class re-scoring of candidate masks")
    parser.add_argument("--device_map_auto", action="store_true", help="Use device_map='auto' for model loading. Overrides --qwen_device if set.")

    args = parser.parse_args()

    if args.eval:
        # The run_cli_evaluation function will handle setting the seed with the parsed args
        run_cli_evaluation(args)
    else:
        run_streamlit_app(args)