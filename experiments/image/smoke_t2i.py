#!/usr/bin/env python3
"""Smoke test: load Qwen-Image-2.1 from a local directory and generate once.

Prints load time, generation time and peak VRAM, which are the three numbers
that decide what an evolution loop over this model can afford. Picks the GPU
with the most free memory unless CUDA_VISIBLE_DEVICES is already set, because
this is a shared machine and the free card changes hour to hour.
"""
import argparse
import os
import subprocess
import time


def freest_gpu():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    rows = [tuple(int(x) for x in l.split(",")) for l in out.strip().splitlines()]
    idx, used, total = max(rows, key=lambda r: r[2] - r[1])
    return idx, total - used


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/Qwen-Image-2.1")
ap.add_argument("--prompt", default='A neon shop sign that reads "QWEN IMAGE 2.1", '
                'rainy night, reflections on wet pavement')
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--size", type=int, nargs=2, default=[1024, 1024])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="outputs/smoke.png")
args = ap.parse_args()

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    idx, free = freest_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    print(f"  using GPU {idx} ({free} MiB free)")

import torch
from diffusers import QwenImage21Pipeline

t0 = time.time()
pipe = QwenImage21Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
pipe.to("cuda")
print(f"  loaded in {time.time() - t0:.1f}s, "
      f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB resident")

torch.cuda.reset_peak_memory_stats()
t1 = time.time()
img = pipe(prompt=args.prompt, width=args.size[0], height=args.size[1],
           num_inference_steps=args.steps,
           generator=torch.Generator("cuda").manual_seed(args.seed)).images[0]
gen = time.time() - t1
os.makedirs(os.path.dirname(args.out), exist_ok=True)
img.save(args.out)
print(f"  generated {args.size[0]}x{args.size[1]} in {gen:.1f}s "
      f"({gen / args.steps:.2f}s/step), peak "
      f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB -> {args.out}")
