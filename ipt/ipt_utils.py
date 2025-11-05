
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration
from qwen_vl_utils import process_vision_info


from PIL import Image, ImageDraw



def load_qwen_model(model_name):
    
   
    model = None
   
   
    if(model_name.startswith("Qwen2.5-VL")): 
        print("Loading using Qwen2_5_VLForConditionalGeneration")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/"+model_name,
        dtype= torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
        
    elif(model_name.startswith("Qwen3-VL-235B") or model_name.startswith("Qwen3-VL-30B")):
        print("Loading using Qwen3VLMoeForConditionalGeneration")
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            # "Qwen/"+model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
            "Qwen/"+model_name, 
            dtype=torch.bfloat8 if model_name.startswith("Qwen3-VL-235B") else torch.bfloat16, 
            attn_implementation="flash_attention_2", device_map="auto"
        ) 

    elif(model_name.startswith("Qwen3-VL")):
        print("Loading using Qwen3VLForConditionalGeneration")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/"+model_name, dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
        )
        
        # print("Loading using AutoModelForImageTextToText")
        # model = AutoModelForImageTextToText.from_pretrained(
        #                 f"Qwen/{model_name}",
        #                 # trust_remote_code=True,
        #                 dtype=torch.bfloat8 if model_name.startswith("Qwen3-VL-235B") else torch.bfloat16,
        #                 attn_implementation="flash_attention_2",
        #                 device_map="auto"
        #             )
    
    else:
        print("Error: Invalid model name")
        return None, None


    print(f"\n\nLoaded the model with the following config: \n\n{model.config.model_type}\n\n")

    processor = AutoProcessor.from_pretrained("Qwen/"+model_name)
    # processor = AutoProcessor.from_pretrained(
    #                     f"Qwen/{model_name}", 
    #                     # trust_remote_code=True
    #                 )
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



# VQA utils

import numpy as np

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

            # Find the matching key ignoring case
            matched_key = next((key for key in dataset_instructions_json.keys() if key.lower() == class_name.lower()), None)
            if matched_key:
                dataset_instructions = dataset_instructions_json[matched_key]
            else:
                #Throw error
                raise ValueError(f"Class name '{class_name}' not found in dataset instructions JSON keys.")
        
        return dataset_instructions



    def getPrompt(prompt, dataset_instructions_json):

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



# Drawing utils


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



def create_img_with_bbox(original_image, bbox_xywh):
    """Draws a single red bounding box on an image."""
    img_with_bbox = original_image.copy()
    draw = ImageDraw.Draw(img_with_bbox)
    x, y, w, h = bbox_xywh
    bbox_xyxy = [x, y, x + w, y + h]
    draw.rectangle(bbox_xyxy, outline='red', width=3)
    return img_with_bbox