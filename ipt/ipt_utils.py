
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration
from qwen_vl_utils import process_vision_info


from PIL import Image, ImageDraw


import random


def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def get_masked_image_vqa_class_scores(qwen_model, qwen_processor, prompt_list, pil_images: list, class_name_list: list, batch_size: int = 8):
    """
    Scores a batch of images with bounding boxes based on a VQA prompt.
    This function is adapted from GridVQAscores_withSavedSAMProposal_webUI_RefCOCO_officialEval_saveInterimResults_gridWeightedBBox.py
    """
    if not pil_images: return np.array([])
    

    num_cls = len(class_name_list)

    def getPrompt(prompt):
        
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
            cls_token_probs = torch.tensor(cls_token_probs)
            cls_token_probs = cls_token_probs/cls_token_probs.sum()
            print("[{b}] Cls-Tokens Probs:" + str([f"Cls-{cls}: {t}" for cls, t in zip(range(num_cls), cls_token_probs)]))

            all_final_scores.append({'cls_name': class_name_list[cls_token_probs.argmax()], 'cls_prob':cls_token_probs.max()})
        
      
    
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





def run_inference_on_single_image(args, model, processor, image_path, dataset_instructions_json, class_name_list, 
                                #   no_instructions=False, few_shot_examples=None, output_dir="."):
                                    no_instructions=False, few_shot_dict=None, output_dir="."):
    """
    Runs Qwen inference on a single image and parses the output.
    """
    set_seed(args.seed)

    raw_output = ""
    parsed_bboxes = []
    all_few_shot_examples = []

    for class_name in class_name_list: #GRG: This iterates over all categories in the dataset - which can is the right thing to do here - but can penalize results as we expect the model to predict all categories in each image
        
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

        parsed_bboxes_i = parse_qwen_output_to_detections(raw_output_i, [class_name], output_dir=output_dir)

        #For Qwen3-VL, Qwen3-VL's default coordinate system has been changed from the absolute coordinates used in Qwen2.5-VL to relative coordinates ranging from 0 to 1000. (You don't need to calculate the resized_w)
        # Ref: https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb 
        if not args.model_name.startswith("Qwen2.5-VL"):
            input_height = 1000
            input_width = 1000
            assert args.model_name.startswith("Qwen3-VL")

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
       
        # vqa_scores = utils.get_masked_image_vqa_scores(
        #     model, processor, vqa_prompt, vqa_images, batch_size=args.vqa_batch_size
        # )
        if args.class_rescore:
            vqa_dict = get_masked_image_vqa_class_scores(
                model, processor, vqa_prompts, vqa_images, class_name_list, batch_size=args.vqa_batch_size
            )
        else:
            # vqa_scores = utils.get_masked_image_vqa_scores(
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