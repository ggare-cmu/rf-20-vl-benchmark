

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from tqdm import tqdm
import gc
import argparse


import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration
from qwen_vl_utils import process_vision_info

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from vllm import SamplingParams

from PIL import Image, ImageDraw

import os
import json
import time
import re

import numpy as np

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
   
    assert model_name == "Qwen3-VL-235B-A22B-Instruct-FP8", "Error: Only Qwen3-VL-235B-A22B-Instruct-FP8 is supported in this setup."
   
    if(model_name.startswith("Qwen2.5-VL")): 
        print("Loading using Qwen2_5_VLForConditionalGeneration")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/"+model_name,
        dtype= torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
        
    elif(model_name == "Qwen3-VL-2B-Instruct-FP8" or model_name == "Qwen3-VL-235B-A22B-Instruct-FP8"):
        from vllm import LLM
        model = LLM(
            model="Qwen/"+model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.80,
            enforce_eager=False,
            enable_expert_parallel = True,
            # max_model_len=700,
            tensor_parallel_size=torch.cuda.device_count(),
            seed=0
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


# Model inference utils



def model_generate(messages, model, processor):
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text_input], images=image_inputs, padding=True, return_tensors="pt").to(model.device)

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

        
    return output_text, inputs



# def model_generate_with_scores(conversations, model, processor, max_new_tokens=2):
def model_generate_with_scores(conversations, model, processor, max_new_tokens=1):
    # conversations is a list of message lists

    # Prepare inputs for the model
    text_input = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(conversations)
    inputs = processor(text=text_input, images=image_inputs, padding=True, return_tensors="pt").to(model.device)
    
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
            logprobs=1,   # get top-5 logprobs per token
            stop_token_ids=[],
        )
        outputs = model.generate(inputs, sampling_params = sampling_params)
        # for i, output in enumerate(outputs):
        #     output_text = output.outputs[0].text

    # # Generate outputs
    # with torch.inference_mode():
    #     outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, output_scores=True, return_dict_in_generate=True)
    
    
    return outputs



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
    
        
    yes_token_id = qwen_processor.tokenizer.encode("Yes")[0]
    no_token_id = qwen_processor.tokenizer.encode("No")[0]

    all_final_scores = []
    # Process images in batches
    for i in range(0, len(pil_images), batch_size):
        batch_pil_images = pil_images[i:i + batch_size]
        batch_prompts = prompt_list[i:i + batch_size]
        
        # Create conversations for the batch
        conversations = [[{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": getPrompt(prompt, dataset_instructions_json)}]}] for img, prompt in zip(batch_pil_images, batch_prompts)]
        
        # Generate outputs with scores
        outputs = model_generate_with_scores(conversations, qwen_model, qwen_processor)

        # # Calculate 'Yes' probability
        # scores = outputs.scores[0]
        # probs = torch.nn.functional.softmax(scores, dim=-1)
        
        # yes_probs, no_probs = probs[:, yes_token_id], probs[:, no_token_id]
        # batch_scores = (yes_probs / (yes_probs + no_probs + 1e-18)).cpu().numpy()
        # all_final_scores.extend(batch_scores.tolist())

        for output in outputs:
            token_logprobs = output.outputs[0].logprobs  # list[dict]

            # Initialize scores
            yes_logprob = None
            no_logprob = None

            # Look through generated tokens and their top_logprobs
            for token_info in token_logprobs:
                top_logprobs = token_info["top_logprobs"]
                if "Yes" in top_logprobs:
                    yes_logprob = top_logprobs["Yes"]
                if "No" in top_logprobs:
                    no_logprob = top_logprobs["No"]

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


# def get_image_textbbox_vqa_scores_with_instructions(qwen_model, qwen_processor, dataset_instructions_json, image, det_bboxes, batch_size: int = 8):
#     """
#     Scores a batch of images with bounding boxes based on a VQA prompt.
#     This function is adapted from GridVQAscores_withSavedSAMProposal_webUI_RefCOCO_officialEval_saveInterimResults_gridWeightedBBox.py
#     """
#     # if not pil_images: return np.array([])
    
#     def getDatasetInstructions(dataset_instructions_json, class_name):
#         if class_name in dataset_instructions_json:       
#             dataset_instructions = dataset_instructions_json[class_name]
#         else:

#             # Find the matching key ignoring case
#             matched_key = next((key for key in dataset_instructions_json.keys() if key.lower() == class_name.lower()), None)
#             if matched_key:
#                 dataset_instructions = dataset_instructions_json[matched_key]
#             else:
#                 #Throw error
#                 raise ValueError(f"Class name '{class_name}' not found in dataset instructions JSON keys.")
        
#         return dataset_instructions



#     def getPrompt(det, dataset_instructions_json):

#         # question = f"""
#         #             You are given the definition of the class '{det['category_name']}': 
#         #             {getDatasetInstructions(dataset_instructions_json, det['category_name'])}

#         #             Carefully examine the image and the bounding box defined as bbox = {det['bbox_model_xyxy']}.

#         #             Question: Does this bounding box fully contain exactly one instance of the object '{det['category_name']}' — meaning:
#         #             1. The object is entirely inside the box (no visible part extends outside), and 
#         #             2. No other object (of any class) is present within the same box.

#         #             Please answer strictly with 'Yes' or 'No'.
#         # """

#         question = f"""
#             Given the '{det['category_name']}' class defined as follows: {getDatasetInstructions(dataset_instructions_json, det['category_name'])}

#             Is the main subject or object being referred to as: '{det['category_name']}' located inside the bounding box defined as bbox = {det['bbox_model_xyxy']} in the image? Please answer Yes or No. Note: The object should be entirely inside the bounding box, with no part outside, and it must be the only object present inside - no other objects should appear within the box.
#         """

#         return question
        
#     yes_token_id = qwen_processor.tokenizer.encode("Yes")[0]
#     no_token_id = qwen_processor.tokenizer.encode("No")[0]

#     all_final_scores = []
#     # Process images in batches
#     for i in range(0, len(det_bboxes), batch_size):
#         batch_det_bboxes = det_bboxes[i:i + batch_size]
        
#         # Create conversations for the batch
#         conversations = [[{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": getPrompt(det, dataset_instructions_json)}]}] for det in batch_det_bboxes]
        
#         # Generate outputs with scores
#         outputs = model_generate_with_scores(conversations, qwen_model, qwen_processor)

#         # Calculate 'Yes' probability
#         scores = outputs.scores[0]
#         probs = torch.nn.functional.softmax(scores, dim=-1)
        
#         yes_probs, no_probs = probs[:, yes_token_id], probs[:, no_token_id]
#         batch_scores = (yes_probs / (yes_probs + no_probs + 1e-18)).cpu().numpy()
#         all_final_scores.extend(batch_scores.tolist())
    
#     return np.array(all_final_scores)




# def get_image_textbbox_batched_vqa_scores_with_instructions(qwen_model, qwen_processor, dataset_instructions_json, image, det_bboxes, batch_size: int = 10):
#     """
#     Scores a batch of images with bounding boxes based on a VQA prompt.
#     This function is adapted from GridVQAscores_withSavedSAMProposal_webUI_RefCOCO_officialEval_saveInterimResults_gridWeightedBBox.py
#     """
#     # if not pil_images: return np.array([])
  

#     # def getPrompt(det, class_name_list, dataset_instructions_json):

#     #     # question = f"""
#     #     #             You are given the definition of the class '{det['category_name']}': 
#     #     #             {getDatasetInstructions(dataset_instructions_json, det['category_name'])}

#     #     #             Carefully examine the image and the bounding box defined as bbox = {det['bbox_model_xyxy']}.

#     #     #             Question: Does this bounding box fully contain exactly one instance of the object '{det['category_name']}' — meaning:
#     #     #             1. The object is entirely inside the box (no visible part extends outside), and 
#     #     #             2. No other object (of any class) is present within the same box.

#     #     #             Please answer strictly with 'Yes' or 'No'.
#     #     # """


#     #     def promptTemplate(det):
#     #         template = f"Is the main subject or object being referred to as: '{det['category_name']}' located inside the bounding box defined as bbox = {det['bbox_model_xyxy']} in the image? Please answer Yes or No."
#     #         return template
        
#     #     question = f"""
#     #         Given the following class names: {class_name_list}, which are defined as follows: {dataset_instructions_json}

#     #         Answer the following questions and provide output as a python json dictionary with keys 'Question-idx' and values as (Yes/No):
#     #     """


#     #     for idx, det in enumerate(det_bboxes):
#     #         question += f"Question-{idx}: {promptTemplate(det)} \n"
            

#     def getPrompt(batch_det_bboxes, class_name_list, dataset_instructions_json):


#         def questionTemplate(det, idx):
#             return (
#                 f"Question-{idx}: For the object class '{det['category_name']}', "
#                 f"is there exactly one complete instance of this object located entirely within "
#                 f"the bounding box defined as bbox = {det['bbox_model_xyxy']}? "
#                 "The object must be fully contained (no part outside the box) and no other objects should appear inside. "
#                 "Please answer strictly with 'Yes' or 'No'."
#             )

#         question = f"""
#             You are given the following class names: {class_name_list}, each defined as follows: {dataset_instructions_json}

#             Carefully examine the image and answer each question below.

#             Provide your final response as a valid Python JSON dictionary where:
#             - Each key is 'Question-<idx>'
#             - Each value is either 'Yes' or 'No'

#             Example output format:
#             {{
#             '0': 'Yes',
#             '1': 'No',
#             '2': 'Yes'
#             }}

#             Questions:
#         """

#         for idx, det in enumerate(batch_det_bboxes):
#             question += questionTemplate(det, idx) + "\n"

#         return question


#     batch_size = 10
#     print(f"Using batch size: {batch_size} for VQA bbox scoring.")

#     yes_token_id = qwen_processor.tokenizer.encode("Yes")[0]
#     no_token_id = qwen_processor.tokenizer.encode("No")[0]

#     all_final_scores = []
#     # Process images in batches
#     for i in range(0, len(det_bboxes), batch_size):
#         batch_det_bboxes = det_bboxes[i:i + batch_size]
        
#         det_classes = set([det['category_name'] for det in batch_det_bboxes])

#         # Create conversations for the batch
#         messages = [{"role": "user", "content": [
#                         {"type": "image", "image": image}, 
#                         {"type": "text", "text": getPrompt(batch_det_bboxes, det_classes, dataset_instructions_json)}
#                         ]
#                     }]
        
#         # Generate outputs with scores
#         outputs = model_generate_with_scores(messages, qwen_model, qwen_processor, max_new_tokens=batch_size*100)

#         predicted_tokens = [qwen_processor.tokenizer.decode(torch.argmax(outputs.scores[i], dim=-1)) for i in range(len(outputs.scores))]
#         print(f"[{i}] Predicted tokens: {predicted_tokens}")


#         def getAnwserIndex(predicted_tokens):

#             question_answer_map = {}

#             current_question_idx = -1
#             current_question = ''
#             answer_index = -1
#             answer = ''
#             for idx, token in enumerate(predicted_tokens):
#                 #Find question idx
#                 if token.isdigit():
#                     # current_question_idx = int(token)
#                     current_question_idx = idx
#                     current_question = f"Question-{token}"
                
#                 if token.lower() in ['yes', 'no'] and current_question_idx != -1:
#                     answer = token
#                     answer_index = idx

                    
#                     question_answer_map[current_question] = {
#                         'answer': answer,
#                         'answer_index': answer_index,
#                         'question': current_question,
#                         'question_idx': current_question_idx
#                     }

#                     #Reset for next question
#                     current_question_idx = -1
#                     current_question = ''
#                     answer_index = -1
#                     answer = ''

#             return question_answer_map


            

#         question_answer_map = getAnwserIndex(predicted_tokens)
#         print(f"[{i}] Found question_answer_map: {question_answer_map}")

#         if len(question_answer_map) != len(batch_det_bboxes):
#             print(f"\n\n******\n[{i}] Warning! Number of answers ({len(question_answer_map)}) does not match number of bboxes ({len(batch_det_bboxes)})!\n******\n\n")
        

#         for b, det in enumerate(batch_det_bboxes):
#             qa_key = f"Question-{b}"
#             if qa_key not in question_answer_map:
#                 print(f"\n\n******\n[{i}] Warning! {qa_key} not found in question_answer_map!\n******\n\n")
#                 all_final_scores.append(0.0)
#                 continue

#             answer_info = question_answer_map[qa_key]
#             answer_index = answer_info['answer_index']

#             # Calculate 'Yes' probability
#             scores = outputs.scores[answer_index]
#             probs = torch.nn.functional.softmax(scores, dim=-1)
            
            
#             yes_probs, no_probs = probs[:, yes_token_id], probs[:, no_token_id]
#             score = (yes_probs / (yes_probs + no_probs + 1e-18)).cpu().numpy()
#             all_final_scores.append(score.tolist()[0])

    
#     return np.array(all_final_scores)




# def get_masked_image_vqa_scores(qwen_model, qwen_processor, prompt_list, pil_images: list, batch_size: int = 8):
#     """
#     Scores a batch of images with bounding boxes based on a VQA prompt.
#     This function is adapted from GridVQAscores_withSavedSAMProposal_webUI_RefCOCO_officialEval_saveInterimResults_gridWeightedBBox.py
#     """
#     if not pil_images: return np.array([])
    
#     def getPrompt(prompt):
#         # question = f"Is the main subject or object being referred to in this sentence: '{prompt}' located inside the red bounding box in the image? Please answer yes or no. Note: The object should be entirely inside the bounding box, with no part outside, and it must be the only object present inside - no other objects should appear within the box."
#         question = f"Is the main subject or object being referred to as: '{prompt}' located inside the red bounding box in the image? Please answer Yes or No. Note: The object should be entirely inside the bounding box, with no part outside, and it must be the only object present inside - no other objects should appear within the box."
#         return question

        
#     yes_token_id = qwen_processor.tokenizer.encode("Yes")[0]
#     no_token_id = qwen_processor.tokenizer.encode("No")[0]
    
#     all_final_scores = []
#     # Process images in batches
#     for i in range(0, len(pil_images), batch_size):
#         batch_pil_images = pil_images[i:i + batch_size]
#         batch_prompts = prompt_list[i:i + batch_size]
        
#         # Create conversations for the batch
#         conversations = [[{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": getPrompt(prompt)}]}] for img, prompt in zip(batch_pil_images, batch_prompts)]
        
#         # Generate outputs with scores
#         outputs = model_generate_with_scores(conversations, qwen_model, qwen_processor)

#         # Calculate 'Yes' probability
#         scores = outputs.scores[0]
#         probs = torch.nn.functional.softmax(scores, dim=-1)
        
#         yes_probs, no_probs = probs[:, yes_token_id], probs[:, no_token_id]
#         batch_scores = (yes_probs / (yes_probs + no_probs + 1e-18)).cpu().numpy()
#         all_final_scores.extend(batch_scores.tolist())
    
#     return np.array(all_final_scores)







# import string
# def getClsIndex(predicted_tokens, num_cls):

#     #Check for col - which is a character - but we muct also include cases like '[A' which is a single token
#     cls_char = [f"{cls}" for cls in range(num_cls)]

#     cls_index = np.ones(len(predicted_tokens))
    
#     default_cls_idx = 1
    
#     #Direct Match
#     dir_cls_match = np.array([t in cls_char for t in predicted_tokens])
    
#     found_cls = False

#     if np.any(dir_cls_match):
#         found_cls = True
#         cls_index = dir_cls_match


#     #Find any char or digit containing token
#     if not found_cls:
#         cls_index[np.where(np.array(predicted_tokens) == '<|im_end|>')] = 0

#         # digit_index = [np.any([c.isdigit() and c not in string.punctuation for c in s]) for s in predicted_tokens]
#         alpha_index = [np.any([c.isalpha() and c not in string.punctuation for c in s]) for s in predicted_tokens]

#         # cls_index = np.logical_and(cls_index, digit_index)
#         cls_index = np.logical_and(cls_index, alpha_index)

#         if np.any(cls_index):
#             found_cls = True


#     #Handling default fallback
#     if not found_cls:
#         cls_index = np.ones(len(predicted_tokens))
#         cls_index[default_cls_idx] = 1


#     return cls_index


# def get_masked_image_vqa_class_scores(qwen_model, qwen_processor, prompt_list, pil_images: list, class_name_list: list, batch_size: int = 8):
#     """
#     Scores a batch of images with bounding boxes based on a VQA prompt.
#     This function is adapted from GridVQAscores_withSavedSAMProposal_webUI_RefCOCO_officialEval_saveInterimResults_gridWeightedBBox.py
#     """
#     if not pil_images: return np.array([])
    

#     num_cls = len(class_name_list)

#     def getPrompt(prompt):
        
#         # class_options = [f"[{chr(ord('A') + i)}]: {c}" for i, c in enumerate(class_name_list)]
#         class_options = [f"${chr(ord('A') + i)}$: {c}" for i, c in enumerate(class_name_list)]
#         class_options_str = ", ".join(class_options)
#         # example_class_token = f"[{chr(ord('A'))}]"
#         example_class_token = f"${chr(ord('A'))}$"
        
#         # question = f"Identify which class the subject or object inside the red bounding box belongs to from the following options: {class_options_str}. Respond only with the class index letter. For example, if the class is {class_name_list[0]}, output {example_class_token}."
#         question = f"Give the class name index the subject or object located inside the red bounding box in the image better relates to from the following: {class_options_str}? Please only answer using the class name index number. Ex for class name: {class_name_list[0]}, output: {example_class_token}."
#         return question

#     all_final_scores = []
#     # Process images in batches
#     for i in range(0, len(pil_images), batch_size):
#         batch_pil_images = pil_images[i:i + batch_size]
#         batch_prompts = prompt_list[i:i + batch_size]
        
#         # Create conversations for the batch
#         conversations = [[{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": getPrompt(prompt)}]}] for img, prompt in zip(batch_pil_images, batch_prompts)]
 

#         # Generate outputs with scores
#         outputs = model_generate_with_scores(conversations, qwen_model, qwen_processor, max_new_tokens=5)

                
#         for b in range(len(batch_pil_images)):
#             predicted_tokens = [qwen_processor.tokenizer.decode(torch.argmax(outputs.scores[i][b], dim=-1)) for i in range(len(outputs.scores))]
#             print(f"[{b}] Predicted tokens: {predicted_tokens}")
    
#             cls_index = getClsIndex(predicted_tokens, num_cls)
#             print(f"[{b}] Found cls_index: {cls_index}")
            
#             # We take the mean for all found tokens 
#             cls_scores = torch.concat([outputs.scores[i][b].unsqueeze(0) for i in range(len(outputs.scores)) if cls_index[i] == 1], dim = 0).mean(dim = 0).unsqueeze(0)
            
#             # Get token IDs for 'A', 'B', 'C', etc.
#             cls_token_ids = [qwen_processor.tokenizer.encode(f"{chr(ord('A') + i)}")[0] for i in range(num_cls)]
#             print("[{b}] Cls-Tokens:" + str([f"Cls-{cls}: {t}" for cls, t in zip(range(num_cls), cls_token_ids)]))
        
#             # Print the next predicted token 
#             predicted_cls_token = qwen_processor.tokenizer.decode(torch.argmax(cls_scores, dim=-1))
#             print("[{b}] Cls predicted token:", predicted_cls_token)
            
#             cls_probs = torch.nn.functional.softmax(cls_scores, dim=-1)
            
#             if len(cls_token_ids) != len(set(cls_token_ids)):
#                 print(f"\n\n******\n[{b}] Error! Cls Token ids aren't unique: {np.unique(cls_token_ids, return_counts = True)}\n******\n\n")

#             cls_token_probs = [cls_probs[:, cls] for cls in cls_token_ids]
#             print("[{b}] Cls-Tokens Probs:" + str([f"Cls-{cls}: {t}" for cls, t in zip(range(num_cls), cls_token_probs)]))

#             #Normalize the col & row token probs - to avoid row/col domination
#             cls_token_probs = torch.tensor(cls_token_probs)
#             cls_token_probs = cls_token_probs/cls_token_probs.sum()
#             print("[{b}] Cls-Tokens Probs:" + str([f"Cls-{cls}: {t}" for cls, t in zip(range(num_cls), cls_token_probs)]))

#             all_final_scores.append({'cls_name': class_name_list[cls_token_probs.argmax()], 'cls_prob':cls_token_probs.max()})
        
      
    
#     return all_final_scores


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




# def run_qwen_inference(args, model, processor, image_path, dataset_instructions, class_name, no_instructions=False, few_shot_examples=None):
def run_qwen_inference(args, model, processor, image, dataset_instructions, class_name, no_instructions=False, few_shot_examples=None):
    """
    Given a model, processor, local image path, instructions (from README), 
    and the current image's filename, run Qwen2.5-VL and return the raw text output.
    """
    set_seed(args.seed)

    # #image = Image.open(image_path).convert("RGB")
    # if image is None:
    #     image = Image.open(image_path).convert("RGB")


    if no_instructions:

        prompt_text = (
        # *** Best so far ***
 
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
        
        messages = [
            {
                "role": "user",
                "content": [
        
                    {"type": "text", "text": ( 
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
                    {"type": "image", "image": image},
                ],
            }
        ]
    else:
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
    # input_height = inputs['image_grid_thw'][0][1]*14
    # input_width = inputs['image_grid_thw'][0][2]*14
    input_height = 1000
    input_width = 1000

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




def run_inference_on_single_image(args, model, processor, image_path, dataset_instructions_json, class_name_list, 
                                #   no_instructions=False, few_shot_examples=None, output_dir="."):
                                    no_instructions=False, few_shot_dict=None, output_dir="."):
    """
    Runs Qwen inference on a single image and parses the output.
    """
    set_seed(args.seed)

    original_image = Image.open(image_path).convert("RGB")


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
            # image_path=image_path,
            image=original_image,
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
       
        # vqa_scores = get_masked_image_vqa_scores(
        #     model, processor, vqa_prompt, vqa_images, batch_size=args.vqa_batch_size
        # )
        if args.class_rescore:
            vqa_dict = get_masked_image_vqa_class_scores(
                model, processor, parsed_bboxes, vqa_images, class_name_list, batch_size=args.vqa_batch_size
            )
        else:
            # vqa_scores = get_masked_image_vqa_scores(
            #     model, processor, vqa_prompts, vqa_images, batch_size=args.vqa_batch_size
            # )
            vqa_scores = get_masked_image_vqa_scores_with_instructions(
                model, processor, dataset_instructions_json, vqa_prompts, vqa_images, batch_size=args.vqa_batch_size
            )
            # vqa_scores = get_image_textbbox_vqa_scores_with_instructions(
            #     model, processor, dataset_instructions_json, original_image, parsed_bboxes, batch_size=args.vqa_batch_size
            # )
            # vqa_scores = get_image_textbbox_batched_vqa_scores_with_instructions(
            #     model, processor, dataset_instructions_json, original_image, parsed_bboxes, batch_size=args.vqa_batch_size
            # )
        
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


# Formating utils

# def format_coco_metrics(stats):
#     """
#     Formats the 12 COCO metrics into a readable string, similar to coco_eval.summarize().
#     """
#     metric_names = [
#         ("Average Precision", "(AP)", "IoU=0.50:0.95", "all", 100),
#         ("Average Precision", "(AP)", "IoU=0.50", "all", 100),
#         ("Average Precision", "(AP)", "IoU=0.75", "all", 100),
#         ("Average Precision", "(AP)", "IoU=0.50:0.95", "small", 100),
#         ("Average Precision", "(AP)", "IoU=0.50:0.95", "medium", 100),
#         ("Average Precision", "(AP)", "IoU=0.50:0.95", "large", 100),
#         ("Average Recall", "(AR)", "IoU=0.50:0.95", "all", 1),
#         ("Average Recall", "(AR)", "IoU=0.50:0.95", "all", 10),
#         ("Average Recall", "(AR)", "IoU=0.50:0.95", "all", 100),
#         ("Average Recall", "(AR)", "IoU=0.50:0.95", "small", 100),
#         ("Average Recall", "(AR)", "IoU=0.50:0.95", "medium", 100),
#         ("Average Recall", "(AR)", "IoU=0.50:0.95", "large", 100),
#     ]
    
#     formatted_string = ""
#     for i, (title, type_str, iou, area, max_dets) in enumerate(metric_names):
#         formatted_string += f"{title:<18} {type_str:<5} @[ IoU={iou:<9} | area={area:>6s} | maxDets={max_dets:>3d} ] = {stats[i]:0.3f}\n"
        
#     return formatted_string




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