# Qwen3 VL 32B Implementation Plan for TorchTitan

Implement Qwen3 VL 32B as a new experiment in TorchTitan, a multimodal vision-language model with DeepStack, MRoPE, and Q/K normalization.

## Motivation

Qwen3 VL provides a strong foundation for **computer use and agentic applications**. The model's ability to process screenshots and UI elements makes it well-suited for:

- **RL for computer use** - Training agents to interact with GUIs via reinforcement learning
- **Agentic workflows** - Multi-step tasks requiring visual understanding of application state
- **UI automation** - Grounding actions to screen coordinates and elements

This work supports asks from Yutori for VLM capabilities in RL training pipelines.

---

## Architecture Summary

| Feature | Description |
|---------|-------------|
| **DeepStack** | Multi-level vision feature extraction at layers [8, 16, 24], injected into early text layers |
| **MRoPE** | Multi-dimensional RoPE with interleaved t/h/w sections [24, 20, 20] |
| **Q/K Normalization** | RMSNorm on Q and K after projection, before RoPE |
| **GQA** | 8:1 ratio for 32B (64 attention heads, 8 KV heads) |
| **Patch Merger** | 2x2 spatial merge, reduces sequence length by 4x |

---

## MVP: Images + FSDP

**Goal:** Train Qwen3 VL on image-text data with FSDP parallelism.

### Scope
- Images only (video support deferred)
- FSDP parallelism only (TP/CP deferred)
- Max sequence length: 4096

### Deliverables

1. **Model Implementation**
   - Vision encoder (Conv2D patches, 2D RoPE, DeepStack)
   - Text decoder (MRoPE, Q/K norm, GQA)
   - Patch merger with spatial merge

2. **HuggingFace Weight Loading** *(P0)*
   - State dict adapter for HF ↔ TorchTitan conversion
   - Validate forward pass matches HF reference implementation
   - Test with Qwen3-VL-8B weights

3. **Dataset Integration**
   - Qwen3 VL uses different special tokens than existing VLM:
     - VLM: `<|begin_of_image|>` / `<|end_of_image|>`
     - Qwen3 VL: `<|vision_start|>` / `<|vision_end|>`
   - Options:
     1. **Adapt existing VLM dataloader** - Map Qwen3 VL tokens to VLM interface
     2. **Create Qwen3 VL specific processor** - Fork `mm_datasets.py` with Qwen3 tokens
     3. **Refactor to common interface** - Abstract SpecialTokens protocol
   - Can reuse: image processing, packing, collation logic
   - Supported datasets: `cc12m`, `obelics`, or custom HF datasets

4. **FSDP Parallelization**
   - Per-layer sharding for encoder and text model
   - Activation checkpointing

5. **Training Configs**
   - Debug model (small, for testing)
   - 8B and 32B production configs

### Testing for Correctness

**1. Weight Loading Validation (P0)**
```python
# Load HF model
from transformers import Qwen3VLForConditionalGeneration
hf_model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B")

# Load TorchTitan model with converted weights
tt_model = Qwen3VLTransformer(model_args)
tt_model.load_state_dict(adapter.from_hf(hf_model.state_dict()))

# Compare forward pass
with torch.no_grad():
    hf_out = hf_model(input_ids, pixel_values, ...)
    tt_out = tt_model(input_ids, pixel_values, ...)

assert torch.allclose(hf_out.logits, tt_out, atol=1e-4)
```

**2. Component Unit Tests**
- MRoPE: Verify position encoding matches HF `apply_multimodal_rotary_pos_emb`
- Q/K Norm: Check normalization applied after projection, before RoPE
- DeepStack: Validate feature shapes at extraction layers [8, 16, 24]
- Patch Merger: Verify 2x2 spatial merge produces correct output dimensions

**3. Gradient Flow Test**
```python
# Verify gradients flow through all components
loss = model(tokens, pixel_values, ...).sum()
loss.backward()

for name, param in model.named_parameters():
    assert param.grad is not None, f"No gradient for {name}"
    assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
```

**4. Multi-GPU FSDP Test**
```bash
# Run 4-GPU FSDP test
torchrun --nproc_per_node=4 -m torchtitan \
  --job.config_file torchtitan/experiments/qwen3_vl/train_configs/debug_model.toml

# Verify: Loss decreases, no NCCL errors, checkpointing works
```

**5. Loss Sanity Check**
- Initial loss should be ~ln(vocab_size) ≈ 11.9 for random init
- Loss should decrease over first 100 steps
- Compare loss curve with HF training on same data

---

## Generation for RL

Two options for generation/rollouts in RL training:

### Option 1: vLLM Native (Recommended for MVP)

vLLM has full Qwen3 VL support (`vllm/model_executor/models/qwen3_vl.py`):

```python
class Qwen3VLForConditionalGeneration(
    SupportsMultiModal,  # Images + video
    SupportsLoRA,
    SupportsPP,          # Pipeline parallel
    SupportsMRoPE,       # Multi-dimensional RoPE
)
```

**Features:** PagedAttention, continuous batching, TP/PP, LoRA, image + video

**Pros:**
- Production-ready, no development needed
- Optimized inference performance

**Cons:**
- Separate model from TorchTitan → weight conversion between train/inference
- Potential numerical differences (may affect RL stability)

### Option 2: Unified TorchTitan Wrapper (Future)

Extend `TorchTitanVLLMModelWrapper` (in `torchtitan/experiments/rl/unified/`) to support VLMs:

1. Set `supports_multimodal = True`
2. Handle vision inputs (`pixel_values`, `grid_thw`) in forward
3. Register `Qwen3VLTorchTitan` with vLLM model registry

**Pros:**
- Same model weights for train and inference (no conversion)
- Numerical consistency (batch-invariant backward)
- Tighter integration with TorchTitan training loop

**Cons:**
- Development effort to extend wrapper for multimodal

### Recommendation

| Phase | Approach | Rationale |
|-------|----------|-----------|
| **MVP** | vLLM native | Already works, unblocks RL experiments |
| **Later** | Unified wrapper | If numerical issues arise or tighter integration needed |

---

## M1: Tensor Parallelism (TP)

**Goal:** Enable model parallelism for larger batch sizes and memory efficiency.

### Deliverables
- Column-wise sharding: Q, K, V, gate_proj, up_proj
- Row-wise sharding: O, down_proj
- Vocabulary embedding sharding
- Vision encoder TP support

### Testing
- Compare loss curve: TP=2 vs TP=1 (should match within tolerance)
- Verify weight sharding across ranks
- Memory usage validation

---

## M2: Context Parallelism (CP)

**Goal:** Enable longer sequences by distributing context across GPUs.

### Deliverables
- Ring attention for distributed KV cache
- Vision-text boundary handling (ensure vision features available on all ranks)
- Validate with sequences > 4096

### Testing
- Compare outputs: CP=2 vs CP=1 on same sequence
- Verify correct attention masking across ranks
- Test with long sequences (8K, 16K tokens)

---

## Future Work

- **Extended Context Length** - RoPE scaling (NTK-aware or YaRN), MRoPE adaptation, target 32K+ tokens
- **Video Support** - Conv3D patch embedding, temporal dimension handling (t > 1), video tokens
- **Pipeline Parallelism (PP)** - DeepStack complicates stage boundaries; vision features from intermediate layers must reach early text layers across stages

---

## Strategic Consideration: Native vs Transformers Backend

New SOTA vision models will continue to emerge. TorchTitan has two approaches for model support:

### Option A: Native TorchTitan Implementation (Current Approach)

**Pros:**
- Higher MFU (Model FLOPs Utilization)
- Full control for custom features (e.g., DeepStack injection)
- Bitwise reproducibility

**Cons:**
- Development cost per model
- Maintenance burden

### Option B: Transformers Modeling Backend

**Pros:**
- Faster model support (any HF model works)
- Community leverage

**Cons:**
- Lower MFU
- Less control
- VLM support not yet implemented in backend

### Recommendation

| Approach | When to Use |
|----------|-------------|
| **Native (Qwen3 VL)** | High-priority models needing performance or customization |
| **Transformers backend** | Rapid prototyping, simpler models |

**For this proposal:** Native implementation is appropriate because Qwen3 VL is a priority model for RL/computer-use and DeepStack requires custom injection logic.

---

## References

- [Qwen3-VL HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-32B)
- [TorchTitan VLM Experiment](../vlm/)
- [TorchTitan RL Unified](../rl/unified/)
- [TorchTitan Transformers Backend](../transformers_modeling_backend/)
- [Qwen3 Text Model](../../models/qwen3/)
