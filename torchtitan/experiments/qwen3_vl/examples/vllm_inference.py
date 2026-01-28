"""Minimal vLLM inference example for Qwen3 VL."""

from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(
    model="Qwen/Qwen3-VL-8B-Instruct",
    max_model_len=4096,
    max_num_seqs=1,
)

# Sample image (use a URL or local path)
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

# Build prompt with image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": "Describe what's in this image."},
        ],
    }
]

# Generate
outputs = llm.chat(messages, SamplingParams(max_tokens=256, temperature=0.7))
print(outputs[0].outputs[0].text)
