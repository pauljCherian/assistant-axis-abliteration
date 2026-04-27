"""Verify vLLM honors the baked bias on layer 12 down_proj for the steered checkpoint.

Loads /tmp/vllm_test_humorous (already downloaded from HF) in vLLM fp16 (RTX 8000 = compute 7.5),
generates 4 probe prompts, and prints. If outputs show humor (puns, casual tone),
vLLM is using the bias. If outputs look like baseline-generic Llama, vLLM silently dropped the bias
and we MUST abort Stage 4 before user pays.

For comparison, the Stage 3 alpha-pilot at α=3 humorous (HF transformers, also fp16) produced
pun-style jokes ("paws-itive they're on the shelf") and casual tone with slang.
"""
import sys
from vllm import LLM, SamplingParams

import sys
PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vllm_test_humorous"
print(f"Loading {PATH} in vLLM (fp16, gpu_mem_util=0.7)...", flush=True)
llm = LLM(model=PATH, tensor_parallel_size=1, max_model_len=2048,
          dtype="float16", gpu_memory_utilization=0.7, enforce_eager=True)
print("Loaded.", flush=True)

prompts = [
    "Tell me a joke.",
    "Tell me about your day.",
    "What advice do you have for a struggling friend?",
    "Describe the perfect dinner.",
]
sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=200, seed=42)

# Llama-3.1 chat template applied manually
formatted = [
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + q +
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    for q in prompts
]
outs = llm.generate(formatted, sp)
for q, out in zip(prompts, outs):
    print(f"\n--- Q: {q} ---")
    print(f"A: {out.outputs[0].text[:400]}")

print("\n--- TEST DONE ---")
print("Manual verdict needed: do the outputs look HUMOROUS (puns, casual slang, jokes)?")
print("Baseline Llama would give: formal/generic answers, no jokes unless specifically asked.")
