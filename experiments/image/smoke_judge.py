#!/usr/bin/env python3
"""Smoke test: use the generator's own text encoder as a VQA judge.

Qwen-Image-2.1 ships its text encoder as a complete Qwen3-VL: language
model, vision tower, lm_head and processor. So the same weights that
condition the generator can grade its output, with no second download. The
answer is read as a typed decision, yes/no from next-token logits, so the
judge is a Backend in the jev-evolve sense and every invariance check
applies to it.

Caveat, stated up front: a judge that shares weights with the generator's
conditioning is not independent. It is the cheapest first judge, not the
last word.
"""
import argparse
import os
import subprocess
import time


def freest_gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True,
                         text=True).stdout
    rows = [tuple(int(x) for x in l.split(",")) for l in out.strip().splitlines()]
    idx, used, total = max(rows, key=lambda r: r[2] - r[1])
    return idx, total - used


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/Qwen-Image-2.1")
ap.add_argument("--image", default="outputs/smoke.png")
ap.add_argument("--questions", nargs="+", default=[
    "Is there a neon sign in the image?",
    "Does the sign read exactly 'QWEN IMAGE 2.1'?",
    "Is it raining or is the pavement wet?",
    "Is there a giraffe in the image?"])
args = ap.parse_args()

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    idx, free = freest_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    print(f"  using GPU {idx} ({free} MiB free)")

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

t0 = time.time()
proc = AutoProcessor.from_pretrained(os.path.join(args.model, "processor"))
vlm = Qwen3VLForConditionalGeneration.from_pretrained(
    os.path.join(args.model, "text_encoder"), dtype=torch.bfloat16).to("cuda").eval()
print(f"  judge loaded in {time.time() - t0:.1f}s, "
      f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB")

yes_ids = [proc.tokenizer.encode(s, add_special_tokens=False)[0] for s in (" yes", "Yes", " Yes", "yes")]
no_ids = [proc.tokenizer.encode(s, add_special_tokens=False)[0] for s in (" no", "No", " No", "no")]
img = Image.open(args.image).convert("RGB")

for q in args.questions:
    msgs = [{"role": "user", "content": [{"type": "image"},
             {"type": "text", "text": q + " Answer yes or no."}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
    t1 = time.time()
    with torch.no_grad():
        logits = vlm(**inputs).logits[0, -1].float()
    p = torch.softmax(torch.stack([logits[yes_ids].logsumexp(0),
                                   logits[no_ids].logsumexp(0)]), 0)
    print(f"  P(yes)={p[0]:.3f}  {time.time() - t1:.2f}s  {q}")
