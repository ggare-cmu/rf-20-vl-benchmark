
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

        assert len(outputs) == 1, "Error: Expected single output for single input."


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

def main(args, model, processor, image_path, parsed_bboxes, dataset_instructions_json):
   
    original_image = Image.open(image_path).convert("RGB")

    vqa_images = [create_img_with_bbox(original_image, det["bbox"]) for det in parsed_bboxes]
    
    # Get VQA scores for all bboxes in a single batch call
    # We use the category name of the first detection as the prompt for the whole batch,
    # assuming all detections in this context are for the same class.
    # vqa_prompt = parsed_bboxes[0]["category_name"]
    vqa_prompts = [det["category_name"] for det in parsed_bboxes]
    

    vqa_scores = get_masked_image_vqa_scores_with_instructions(
        model, processor, dataset_instructions_json, vqa_prompts, vqa_images, batch_size=args.vqa_batch_size
    )

    return vqa_scores



# Drawing utils


def create_img_with_bbox(original_image, bbox_xywh):
    """Draws a single red bounding box on an image."""
    img_with_bbox = original_image.copy()
    draw = ImageDraw.Draw(img_with_bbox)
    x, y, w, h = bbox_xywh
    bbox_xyxy = [x, y, x + w, y + h]
    draw.rectangle(bbox_xyxy, outline='red', width=3)
    return img_with_bbox