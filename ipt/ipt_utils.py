
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from vllm import LLM, SamplingParams

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from tqdm import tqdm
import gc
import argparse


import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration
from qwen_vl_utils import process_vision_info

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


from PIL import Image, ImageDraw

import json
import time
import re

import numpy as np

import random


from transformers import pipeline




def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)





def load_sigclip_pipeline():
    # ckpt = "google/siglip2-so400m-patch14-384"
    ckpt = "google/siglip2-base-patch16-naflex"
    pipe = pipeline(model=ckpt, task="zero-shot-image-classification")
    return pipe



def load_qwen_model(model_name):
    
   
    model = None
   
    # assert model_name == "Qwen3-VL-235B-A22B-Instruct-FP8", "Error: Only Qwen3-VL-235B-A22B-Instruct-FP8 is supported in this setup."
   
    # if(model_name.startswith("Qwen2.5-VL")): 
    #     print("Loading using Qwen2_5_VLForConditionalGeneration")
    #     model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/"+model_name,
    #     dtype= torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto"
    # )
        
    # elif(model_name == "Qwen3-VL-2B-Instruct-FP8" or model_name == "Qwen3-VL-235B-A22B-Instruct-FP8"):
    
    # dtype = torch.bfloat8 if model_name.startswith("Qwen3-VL-235B-A22B-Instruct-FP8") else torch.bfloat16
    # dtype = "auto" if model_name.startswith("Qwen3-VL-235B-A22B-Instruct-FP8") else torch.bfloat16
    dtype = "auto" if "-FP8" in model_name else torch.bfloat16
    print(f"Loading using LLM class from vLLM with dtype: {dtype}")

    enable_expert_parallel = True if (model_name.startswith("Qwen3-VL-235B-A22B-Instruct-FP8") or model_name.startswith("Qwen3-VL-30B-A3B-Instruct")) else False
    print(f"enable_expert_parallel: {enable_expert_parallel}")

    tensor_parallel_size = 4 if model_name.startswith("Qwen2.5-VL-7B") else torch.cuda.device_count()
    print(f"tensor_parallel_size: {tensor_parallel_size}")

    model = LLM(
        model="Qwen/"+model_name,
        dtype=dtype,
        trust_remote_code=True,
        # gpu_memory_utilization=0.80,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        enable_expert_parallel = enable_expert_parallel,
        # max_model_len=700,
        tensor_parallel_size=tensor_parallel_size,
        seed=0
    )
        
    # elif(model_name.startswith("Qwen3-VL-235B") or model_name.startswith("Qwen3-VL-30B")):
    #     print("Loading using Qwen3VLMoeForConditionalGeneration")
    #     model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    #         # "Qwen/"+model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
    #         "Qwen/"+model_name, 
    #         dtype=torch.bfloat8 if model_name.startswith("Qwen3-VL-235B") else torch.bfloat16, 
    #         attn_implementation="flash_attention_2", device_map="auto"
    #     ) 

    # elif(model_name.startswith("Qwen3-VL")):
    #     print("Loading using Qwen3VLForConditionalGeneration")
    #     model = Qwen3VLForConditionalGeneration.from_pretrained(
    #         "Qwen/"+model_name, dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
    #     )
        
    #     # print("Loading using AutoModelForImageTextToText")
    #     # model = AutoModelForImageTextToText.from_pretrained(
    #     #                 f"Qwen/{model_name}",
    #     #                 # trust_remote_code=True,
    #     #                 dtype=torch.bfloat8 if model_name.startswith("Qwen3-VL-235B") else torch.bfloat16,
    #     #                 attn_implementation="flash_attention_2",
    #     #                 device_map="auto"
    #     #             )
    
    # else:
    #     print("Error: Invalid model name")
    #     return None, None


    # print(f"\n\nLoaded the model with the following config: \n\n{model.config.model_type}\n\n")

    processor = AutoProcessor.from_pretrained("Qwen/"+model_name)
    # processor = AutoProcessor.from_pretrained(
    #                     f"Qwen/{model_name}", 
    #                     # trust_remote_code=True
    #                 )
    # model.eval()



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


# Model inference utils



def model_generate(messages, model, processor):
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    # inputs = processor(text=[text_input], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
    inputs_org = processor(text=[text_input], images=image_inputs, padding=True, return_tensors="pt")

    with torch.no_grad():
        # generated_ids = model.generate(**inputs, max_new_tokens=512)
        # generated_ids = model.generate(**inputs, max_new_tokens=1024)
        output_text = None
        # if(model_name == "Qwen3-VL-2B-Instruct-FP8" or model_name == "Qwen3-VL-235B-A22B-Instruct-FP8"):
            
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        # if video_inputs is not None:
        #     mm_data['video'] = video_inputs

        inputs =  {
            'prompt': text_input,
            'multi_modal_data': mm_data,
        }
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=2048,
            top_k=-1,
            stop_token_ids=[],
        )
        outputs = model.generate(inputs, sampling_params = sampling_params)
        for i, output in enumerate(outputs):
            output_text = output.outputs[0].text

        
    return output_text, inputs_org



# def model_generate_with_scores(conversations, model, processor, max_new_tokens=2):
def model_generate_with_scores(conversations, model, processor, max_new_tokens=1):
# def model_generate_with_scores(conversations, model, processor, max_new_tokens=5):
    # conversations is a list of message lists

    # Prepare inputs for the model
    text_input = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(conversations)
    # inputs = processor(text=text_input, images=image_inputs, padding=True, return_tensors="pt").to(model.device)
    # inputs = processor(text=text_input, images=image_inputs, padding=True, return_tensors="pt")
    
    with torch.no_grad():
        # generated_ids = model.generate(**inputs, max_new_tokens=512)
        # generated_ids = model.generate(**inputs, max_new_tokens=1024)
        # output_text = None
        # if(model_name == "Qwen3-VL-2B-Instruct-FP8" or model_name == "Qwen3-VL-235B-A22B-Instruct-FP8"):
            
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        # if video_inputs is not None:
        #     mm_data['video'] = video_inputs

        inputs =  {
            'prompt': text_input,
            # 'prompt': conversations,
            'multi_modal_data': mm_data,
        }
        sampling_params = SamplingParams(
            temperature=0,
            # max_tokens=2048,
            max_tokens=max_new_tokens,
            top_k=-1,
            # logprobs=max_new_tokens,   # get top-5 logprobs per token
            logprobs=5,   # get top-5 logprobs per token
            stop_token_ids=[],
        )
        outputs = model.generate(inputs, sampling_params = sampling_params)
        # for i, output in enumerate(outputs):
        #     output_text = output.outputs[0].text

    # # Generate outputs
    # with torch.inference_mode():
    #     outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, output_scores=True, return_dict_in_generate=True)
    
    
    return outputs


# Siglip utils

def rescore_with_sigclip(sigclip_pipe, pil_image, candidate_label):
    output = sigclip_pipe(pil_image, candidate_labels=[candidate_label])
    # print(f"SigClip output: {output} for label: {candidate_label}")
    assert len(output) == 1, "Error: SigClip output length is not 1."

    label_score = output[0]['score']

    return label_score


# VQA utils

import numpy as np

def get_masked_image_vqa_scores_with_instructions(qwen_model, qwen_processor, dataset_instructions_json, prompt_list, pil_images: list, batch_size: int = 8):
    """
    Scores a batch of images with bounding boxes based on a VQA prompt.
    This function is adapted from GridVQAscores_withSavedSAMProposal_webUI_RefCOCO_officialEval_saveInterimResults_gridWeightedBBox.py
    """
    # if not pil_images: return np.array([])
    
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
    
        
    # yes_token_id = qwen_processor.tokenizer.encode("Yes")[0]
    # no_token_id = qwen_processor.tokenizer.encode("No")[0]

    all_final_scores = []
    # Process images in batches
    for i in range(0, len(pil_images)):
        img = pil_images[i]
        prompt = prompt_list[i]
        
        # Create conversations for the batch
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": getPrompt(prompt, dataset_instructions_json)}]}]
        
        # Generate outputs with scores
        outputs = model_generate_with_scores(messages, qwen_model, qwen_processor)

        # # Calculate 'Yes' probability
        # scores = outputs.scores[0]
        # probs = torch.nn.functional.softmax(scores, dim=-1)
        
        # yes_probs, no_probs = probs[:, yes_token_id], probs[:, no_token_id]
        # batch_scores = (yes_probs / (yes_probs + no_probs + 1e-18)).cpu().numpy()
        # all_final_scores.extend(batch_scores.tolist())

        # print(f"Len of outputs: {len(outputs)}; outputs: {outputs}")
        # print(f"Len of outputs: {len(outputs)};")
        assert len(outputs) == 1, "Error: Expected single output for single input."


        # for output in outputs:
        #     # Access the generated tokens' logprobs
        #     for completion_output in output.outputs:
        #         # cumulative log probability of the entire generated sequence
        #         cumulative_logprob = completion_output.cumulative_logprob

        #         # detailed logprobs for each token (a list of dictionaries)
        #         token_logprobs = completion_output.logprobs

        #         for i, logprob_dict in enumerate(token_logprobs):
        #             token_id = completion_output.token_ids[i]
        #             # Each logprob_dict maps token IDs to their logprobs for that position
        #             print(f"Token ID: {token_id}, Logprob: {logprob_dict.get(token_id)}")


        # for output in outputs:
        # token_logprobs = outputs.outputs[0].logprobs  # list[dict]
        # token_logprobs = outputs[0].logprobs  # list[dict]
        # token_logprobs = [v for v in outputs[0].outputs[0].logprobs[0].values()][0]  # list[dict]
        token_logprobs = outputs[0].outputs[0].logprobs[0]  # list[dict]
        # print(f"token_logprobs: {token_logprobs}")


        # Initialize scores
        yes_logprob = None
        no_logprob = None

        # Look through generated tokens and their top_logprobs
        for token_id, token_info in token_logprobs.items():
            # top_logprobs = token_info["top_logprobs"]
            logprob = token_info.logprob
            decoded_token = token_info.decoded_token
            # print(f"Decoded token: {decoded_token}: logprob: {logprob}, token_id: {token_id}")

            if "Yes" == decoded_token:
                yes_logprob = logprob
            if yes_logprob is None and "yes" == decoded_token:
                yes_logprob = logprob

            if "No" == decoded_token:
                no_logprob = logprob
            if no_logprob is None and "no" == decoded_token:
                no_logprob = logprob

        # If neither found, skip
        if yes_logprob is None and no_logprob is None:
            all_final_scores.append(-1.0)
            continue
        if yes_logprob is None:
            no_prob = torch.exp(torch.tensor(no_logprob)) if no_logprob is not None else torch.tensor(0.0)
            yes_prob = 1 - no_prob
            score = yes_prob.item()
            all_final_scores.append(score)
            continue

        # Convert from logprobs to probabilities
        yes_prob = torch.exp(torch.tensor(yes_logprob)) if yes_logprob is not None else torch.tensor(0.0)
        no_prob = torch.exp(torch.tensor(no_logprob)) if no_logprob is not None else torch.tensor(0.0)

        # Normalize to get P(Yes)
        score = yes_prob / (yes_prob + no_prob + 1e-18)
        all_final_scores.append(score.item())
    
    return np.array(all_final_scores)



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




def run_qwen_inference(args, model, processor, image, dataset_instructions, class_name):
    """
    Given a model, processor, local image path, instructions (from README), 
    and the current image's filename, run Qwen2.5-VL and return the raw text output.
    """
    set_seed(args.seed)

    # #image = Image.open(image_path).convert("RGB")
    # if image is None:
    #     image = Image.open(image_path).convert("RGB")


    
    # #Baseline-Prompt: Default with instructions 
    # prompt_text = (
    #     f"Locate all of the following objects: {class_name} in the image and output the coordinates in JSON format.\n\nUse the following annotator instructions to improve detection accuracy:\n{dataset_instructions}\n\nReturn a list of items like {{\"bbox_2d\":[x1,y1,x2,y2],\"label\":\"{class_name}\",\"score\":*confidence_score 0-1*}}."

    # )
    
    prompt_text = (
    f"""
        Follow the steps outlined in the pseudo code below on this image for object detection. 
        Use the dataset’s annotator instructions and class name definitions provided here to guide detection and labeling:\n{dataset_instructions}\n"

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

    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image", "image": image},
            ],
        }
    ]
       

    output_text, inputs = model_generate(messages, model, processor)


    #Sample
    # ```json
    #     [
    #         {"bbox_2d": [1145, 563, 1427, 920], "label": "Adult", "score": 0.95}
    #     ]
    # ```

    # #For scaling the bbox coordinates later
    input_height = inputs['image_grid_thw'][0][1]*14
    input_width = inputs['image_grid_thw'][0][2]*14
    # input_height = 1000
    # input_width = 1000

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
    

    Flag_log_output_text = True

    def log_skipped(reason, item, output_text, Flag_log_output_text):

        with open(skipped_log_path, "a") as f:
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reason": reason,
                "item": item,
                "original_output": output_text if Flag_log_output_text else "Not logged",
            }
            f.write(json.dumps(log_entry) + "\n")

        Flag_log_output_text = False
        return Flag_log_output_text


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
            Flag_log_output_text = log_skipped(reason, text_clean, output_text, Flag_log_output_text)
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
                    Flag_log_output_text = log_skipped(reason, obj_str, output_text, Flag_log_output_text)

    # Process each item in data (which should now be a list of dicts)
    for item in data:
        try:
            if not isinstance(item, dict):
                print(f"Skipping item (not a dict): {item}")
                reason = "Skipping item (not a dict)"
                print(f"{reason}: {item}")
                Flag_log_output_text = log_skipped(reason, item, output_text, Flag_log_output_text)
                continue

            bbox_2d = item.get("bbox_2d", [])
            if len(bbox_2d) != 4:
                print(f"Skipping invalid bbox_2d length: {bbox_2d}")
                reason = "Skipping invalid bbox_2d length"
                print(f"{reason}: {bbox_2d}")
                Flag_log_output_text = log_skipped(reason, item, output_text, Flag_log_output_text)
                continue

            # Convert bbox coordinates to floats (or ints as needed)
            try:
                x1, y1, x2, y2 = map(float, bbox_2d)
            except Exception as e:
                print(f"Skipping item due to conversion error: {bbox_2d}, error: {e}")
                reason = f"Skipping item due to conversion error: {e}"
                print(f"{reason}: {bbox_2d}")
                Flag_log_output_text = log_skipped(reason, item, output_text, Flag_log_output_text)
                continue

            if x2 < x1 or y2 < y1:
                print(f"Skipping reversed coords in bbox_2d: {bbox_2d}")
                reason = "Skipping reversed coords in bbox_2d"
                print(f"{reason}: {bbox_2d}")
                Flag_log_output_text = log_skipped(reason, item, output_text, Flag_log_output_text)
                continue
            w = x2 - x1
            h = y2 - y1
            if w == 0 or h == 0:
                print(f"Skipping zero dimension: {bbox_2d}")
                reason = "Skipping zero dimension"
                print(f"{reason}: {bbox_2d}")
                Flag_log_output_text = log_skipped(reason, item, output_text, Flag_log_output_text)
                continue

            label = item.get("label", "unknown")
            
            if not isinstance(label, str):
                label = str(label) # Convert to string if not already

            if label == "unknown":
                print(f"Skipping item (label is unknown): {item}")
                reason = "Skipping item (label is unknown)"
                print(f"{reason}: {item}")
                Flag_log_output_text = log_skipped(reason, item, output_text, Flag_log_output_text)
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
                    Flag_log_output_text = log_skipped(reason, item, output_text, Flag_log_output_text)
                    
            score = float(item.get("score", -1.0))
            if score == -1.0:
                # print(f"Skipping item (score is -1.0): {item}")
                # reason = "Skipping item (score is -1.0)"
                # print(f"{reason}: {item}")
                # log_skipped(reason, item, output_text)
                # continue
                score = 0.5  # Assign default score and continue
                print(f"Score is -1.0 for item: {item}, so assigning default score 0.5 and continuing")
                reason = f"Score is -1.0 for item: {item}, so assigning default score 0.5 and continuing"
                print(f"{reason}: {item}")
                Flag_log_output_text = log_skipped(reason, item, output_text, Flag_log_output_text)

            detections.append({
                "bbox": [x1, y1, w, h],
                "score": score,
                "category_name": label
            })

        except Exception as e:
            print(f"Skipping item due to error: {item}, error: {e}")
            reason = f"Skipping item due to unexpected error: {e}"
            print(f"{reason}: {item}")
            Flag_log_output_text = log_skipped(reason, item, output_text, Flag_log_output_text)

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


def getMaxInputSizeForQwen(width, height, max_dimension=(2880, 1620)):
    """
    Given an image width and height, returns the maximum input size for Qwen model
    while maintaining aspect ratio and fitting within max_dimension.

    # "height": 1440, "width": 2560, 
    # "height": 1396, "width": 2880
    # "height": 1832, "width": 3360, 
    # "height": 1746, "width": 3114,
    # "height": 4800, "width": 6400,
    #Resize if too large
    # max_dimension = (1920, 1080)
    max_dimension = (2880, 1620)
    # max_dimension = (3840, 2160)

    """

    #Need to see if landscape or portrait
    aspect_ratio = width / height
    if aspect_ratio >= 1.0:
        #Landscape
        if width > max_dimension[0]:
            new_width = max_dimension[0]
            new_height = int(new_width / aspect_ratio)
            max_dimension = (new_width, new_height)

        elif height > max_dimension[1]:
            new_height = max_dimension[1]
            new_width = int(new_height * aspect_ratio)
            max_dimension = (new_width, new_height)
    else:
        #Portrait
        max_dimension = (max_dimension[1], max_dimension[0])  #Swap for portrait

        if height > max_dimension[1]:
            new_height = max_dimension[1]
            new_width = int(new_height * aspect_ratio)
            max_dimension = (new_width, new_height)

        elif width > max_dimension[0]:
            new_width = max_dimension[0]
            new_height = int(new_width / aspect_ratio)
            max_dimension = (new_width, new_height)

    return max_dimension


def run_inference_on_single_image(args, model, processor, image_path, dataset_instructions_json, class_name_list, 
                                #   no_instructions=False, few_shot_examples=None, output_dir="."):
                                    output_dir=".", sigclip_pipe=None):
    """
    Runs Qwen inference on a single image and parses the output.
    """
    set_seed(args.seed)

    original_image = Image.open(image_path).convert("RGB")
    #Check image size
    or_width, or_height = original_image.size
    width, height = original_image.size
    print(f"Original image size: {width}x{height}")

    max_dimension = (2880, 1620)
    if width > max_dimension[0] or height > max_dimension[1]:

        # get the appropriate max dimension while maintaining aspect ratio
        max_dimension = getMaxInputSizeForQwen(width, height, max_dimension)
        
        if width > max_dimension[0] or height > max_dimension[1]:

            print(f"Resizing image from {width}x{height} to fit within {max_dimension[0]}x{max_dimension[1]}")
            original_image = original_image.resize(
                max_dimension,
                Image.Resampling.LANCZOS
            )
            width, height = original_image.size
            print(f"Resized image size: {width}x{height}")

    raw_output = ""
    parsed_bboxes = []

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
        

        set_seed(args.seed)
        
        try:
            # raw_output_i = run_qwen_inference(
            raw_output_i, input_width, input_height = run_qwen_inference(
                args,
                model, processor,
                # image_path=image_path,
                image=original_image,
                dataset_instructions=dataset_instructions,
                class_name=class_name,
            )

        except torch.cuda.OutOfMemoryError as e:
            print("⚠️ CUDA OOM encountered. Retrying with downsized image...")

            # Free up GPU memory
            torch.cuda.empty_cache()

            # Downsize image by 50% (you can adjust this factor)
            width, height = original_image.size
            # max_dimension = (1920, 1080)
            resized_image = original_image.resize(
                # (width // 2, height // 2),
                (1920, 1080),
                Image.Resampling.LANCZOS
            )

            try:
                # Retry inference with downsized image
                raw_output_i, input_width, input_height = run_qwen_inference(
                    args,
                    model, processor,
                    image=resized_image,
                    dataset_instructions=dataset_instructions,
                    class_name=class_name,
                )
                print("✅ Retry succeeded with downsized image.")

            except torch.cuda.OutOfMemoryError:
                print("❌ Still OOM after downsizing. Skipping this image.")
                torch.cuda.empty_cache()
                raw_output_i, input_width, input_height = '', None, None


        except Exception as e:
            print(f"❌ Unexpected error during inference: {e}")
            # raw_output_i, input_width, input_height = '', None, None

            print("Retrying with downsized image...")

            # Free up GPU memory
            torch.cuda.empty_cache()

            # Downsize image by 50% (you can adjust this factor)
            width, height = original_image.size
            # max_dimension = (1920, 1080)
            resized_image = original_image.resize(
                # (width // 2, height // 2),
                (1920, 1080),
                Image.Resampling.LANCZOS
            )

            try:
                # Retry inference with downsized image
                raw_output_i, input_width, input_height = run_qwen_inference(
                    args,
                    model, processor,
                    image=resized_image,
                    dataset_instructions=dataset_instructions,
                    class_name=class_name,
                )
                print("✅ Retry succeeded with downsized image.")

            # except torch.cuda.OutOfMemoryError:
            #     print("❌ Still OOM after downsizing. Skipping this image.")
            #     torch.cuda.empty_cache()
            #     raw_output_i, input_width, input_height = '', None, None
            except Exception as e:
                print(f"❌ Unexpected error during inference: {e}")
                # raw_output_i, input_width, input_height = '', None, None

                print("Retrying with downsized image...")

                # Free up GPU memory
                torch.cuda.empty_cache()

                # Downsize image by 50% (you can adjust this factor)
                width, height = original_image.size
                # max_dimension = (1920, 1080)
                resized_image = original_image.resize(
                    # (width // 2, height // 2),
                    # (1920, 1080),
                    # (1600, 900),
                    (1280, 720),
                    Image.Resampling.LANCZOS
                )

                try:
                    # Retry inference with downsized image
                    raw_output_i, input_width, input_height = run_qwen_inference(
                        args,
                        model, processor,
                        image=resized_image,
                        dataset_instructions=dataset_instructions,
                        class_name=class_name,
                    )
                    print("✅ Retry succeeded with downsized image.")

                # except torch.cuda.OutOfMemoryError:
                #     print("❌ Still OOM after downsizing. Skipping this image.")
                #     torch.cuda.empty_cache()
                #     raw_output_i, input_width, input_height = '', None, None
                except Exception as e:
                    print(f"❌ Unexpected error during inference: {e}")
                    raw_output_i, input_width, input_height = '', None, None
        
        parsed_bboxes_i = parse_qwen_output_to_detections(raw_output_i, [class_name], output_dir=output_dir)

        #For Qwen3-VL, Qwen3-VL's default coordinate system has been changed from the absolute coordinates used in Qwen2.5-VL to relative coordinates ranging from 0 to 1000. (You don't need to calculate the resized_w)
        # Ref: https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb 
        if not args.model_name.startswith("Qwen2.5-VL"):
            input_height = 1000
            input_width = 1000
            assert args.model_name.startswith("Qwen3-VL")

        # Convert normalized coordinates to absolute coordinates - Ref-fix: https://github.com/QwenLM/Qwen3-VL/blob/2f25a646fb0f329647428eb8dacf19293de6f5d4/cookbooks/spatial_understanding.ipynb
        # image = Image.open(image_path).convert("RGB")
        # width, height = original_image.size
        for det in parsed_bboxes_i:
            bbox = det["bbox"]
            x, y, bw, bh = bbox
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            
            # Convert normalized coordinates to absolute coordinates
            abs_y1 = int(y1/input_height * or_height)
            abs_x1 = int(x1/input_width * or_width)
            abs_y2 = int(y2/input_height * or_height)
            abs_x2 = int(x2/input_width * or_width)    

            abs_w = abs_x2 - abs_x1
            abs_h = abs_y2 - abs_y1

            det["bbox"] = [abs_x1, abs_y1, abs_w, abs_h]

            # Also store original (unnormalized) xyxy format
            det["bbox_model_xyxy"] = [x1, y1, x2, y2]

        # Accumulate results for all classes
        parsed_bboxes.extend(parsed_bboxes_i)
        raw_output += f"\n\n--- For class '{class_name}' ---\n{raw_output_i}"
           
    # --- VQA-based Re-scoring ---
    # Create copies for different evaluation paths
    detections_orig_no_nms = [det.copy() for det in parsed_bboxes]
    detections_vqa_no_nms = []

    if args.vqa_rescore and parsed_bboxes:
        # original_image = Image.open(image_path).convert("RGB")
        
        # # Create a list of images, each with one bounding box drawn
        vqa_images = [create_img_with_bbox(original_image, det["bbox"]) for det in parsed_bboxes]
        
        # Get VQA scores for all bboxes in a single batch call
        # We use the category name of the first detection as the prompt for the whole batch,
        # assuming all detections in this context are for the same class.
        # vqa_prompt = parsed_bboxes[0]["category_name"]
        vqa_prompts = [det["category_name"] for det in parsed_bboxes]
       
        try:
            vqa_scores = get_masked_image_vqa_scores_with_instructions(
                model, processor, dataset_instructions_json, vqa_prompts, vqa_images, batch_size=args.vqa_batch_size
            )

        except torch.cuda.OutOfMemoryError as e:
            print("⚠️ CUDA OOM encountered. Retrying with downsized image...")

            # Free up GPU memory
            torch.cuda.empty_cache()

            # Downsize image by 50% (you can adjust this factor)
            width, height = original_image.size
            # max_dimension = (1920, 1080)
            vqa_images_small = []
            for img in vqa_images:
                img_width, img_height = img.size
                if img_width > 1920 or img_height > 1080:
                    img.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
                vqa_images_small.append(img)
            vqa_images = vqa_images_small


            try:
                vqa_scores = get_masked_image_vqa_scores_with_instructions(
                        model, processor, dataset_instructions_json, vqa_prompts, vqa_images, batch_size=args.vqa_batch_size
                )
                print("✅ Retry succeeded with downsized image.")

            except torch.cuda.OutOfMemoryError:
                print("❌ Still OOM after downsizing. Skipping this image.")
                torch.cuda.empty_cache()
                vqa_scores = [-1] * len(parsed_bboxes)

        except Exception as e:
            print(f"❌ Unexpected error during inference: {e}")
            vqa_scores = [-1] * len(parsed_bboxes)
        
        detections_vqa_no_nms = [det.copy() for det in detections_orig_no_nms]

        # Replace original scores with VQA scores
        for i, det in enumerate(detections_vqa_no_nms):
            det["model_score"] = det["score"]  # Keep original model score for reference
            det["vqa_score"] = vqa_scores[i]
            det["score"] = vqa_scores[i] if vqa_scores[i] != -1 else det["score"]

    # --- SigClip-based Re-scoring --- 
    elif args.siglip_rescore and parsed_bboxes:

        detections_vqa_no_nms = [det.copy() for det in detections_orig_no_nms]

        for i, det in enumerate(detections_vqa_no_nms):
            det["model_score"] = det["score"]  # Keep original model score for reference

            #Crop the detected bbox region from the original image
            x, y, w, h = map(int, det["bbox"])
            if w == 0 or h == 0: continue #skip invalid bbox
            cropped_img = original_image.crop((x, y, x + w, y + h))
            # #save cropped image for debugging
            # cropped_img.save(f"cropped_det_{i}.png")

            sigclip_score = rescore_with_sigclip(sigclip_pipe, cropped_img, det["category_name"])

            det["siglip_score"] = sigclip_score
            det["score"] = sigclip_score
        
    else:
        # If not VQA-rescoring, the VQA-based lists are the same as original
        detections_vqa_no_nms = [det.copy() for det in detections_orig_no_nms]
    
    # --- End of VQA-based Re-scoring ---

    # --- Non-Maximum Suppression ---
    detections_orig_with_nms = apply_nms(detections_orig_no_nms, iou_threshold=args.nms_threshold) if args.apply_nms else detections_orig_no_nms
    detections_vqa_with_nms = apply_nms(detections_vqa_no_nms, iou_threshold=args.nms_threshold) if args.apply_nms else detections_vqa_no_nms
    # --- End of NMS ---

    return raw_output, {
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