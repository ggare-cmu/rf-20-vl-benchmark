"""Minimal test: can the Qwen model fit in GPU VRAM via vLLM?"""
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import sys
import torch
from vllm import LLM, SamplingParams

model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-VL-30B-A3B-Instruct"

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print(f"Model: {model_name}")
print()

dtype = "auto" if "-FP8" in model_name else torch.bfloat16
enable_ep = "235B" in model_name or "30B" in model_name

print(f"Loading with dtype={dtype}, expert_parallel={enable_ep} ...")

try:
    model = LLM(
        model="Qwen/" + model_name,
        dtype=dtype,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        enable_expert_parallel=enable_ep,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0,
    )
    print("Model loaded OK!")

    # Quick single-token generation to confirm it actually runs
    outputs = model.generate(
        [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        sampling_params=SamplingParams(max_tokens=1),
    )
    print(f"Test generate OK: {outputs[0].outputs[0].text!r}")

    used = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f"VRAM allocated: {used:.1f} GB, reserved: {reserved:.1f} GB")

except torch.cuda.OutOfMemoryError:
    print("FAILED: Out of GPU memory.")
    sys.exit(1)
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
