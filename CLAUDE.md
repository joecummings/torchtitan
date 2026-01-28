# TorchTitan

PyTorch-native platform for training large generative AI models (LLMs). Provides distributed training with multi-dimensional parallelism.

## Project Structure

```
torchtitan/
├── train.py              # Main entry point - Trainer class and training loop
├── config/               # Configuration system (JobConfig, TOML loading)
├── protocols/            # Extensibility framework (TrainSpec, ModelProtocol)
├── components/           # Reusable training components
│   ├── checkpoint.py     # Distributed checkpoint save/load
│   ├── dataloader.py     # Dataloader interface and C4 dataset
│   ├── loss.py           # Loss computation
│   ├── optimizer.py      # Optimizer builders (AdamW, LAMB, SGD)
│   ├── lr_scheduler.py   # Learning rate scheduling
│   ├── metrics.py        # Training metrics collection
│   └── ft/               # Fault tolerance subsystem
├── distributed/          # Parallelism implementations
│   ├── parallel_dims.py  # ParallelDims mesh layout
│   ├── tensor_parallel.py
│   ├── pipeline_parallel.py
│   ├── context_parallel.py
│   ├── expert_parallel.py
│   └── activation_checkpoint.py
├── models/               # Production model implementations
│   ├── llama3/           # Llama 3.1 (8B, 70B, 405B)
│   ├── llama4/           # Llama 4 (17Bx16E MoE)
│   ├── deepseek_v3/      # DeepSeek-V3 (16B, 671B MoE)
│   ├── qwen3/            # Qwen3 variants
│   └── flux/             # Flux diffusion model
└── experiments/          # Experimental features (self-contained)
    ├── simple_fsdp/      # Simplified FSDP reference
    ├── vlm/              # Vision-Language Models
    └── ...

scripts/                  # Utilities (download, checkpoint conversion, estimation)
tests/                    # Unit and integration tests
docs/                     # Documentation and guides
```

## Key Patterns

**Model structure**: Each model in `models/` follows:
- `model/model.py` - Architecture (pure PyTorch, single-device)
- `model/args.py` - Model configuration
- `infra/parallelize.py` - Parallelization application
- `train_configs/*.toml` - Training configurations
- `__init__.py` - Exports `get_train_spec()`

**Parallelization order**: TP → Activation Checkpointing → torch.compile → FSDP/HSDP

**Configuration**: All training configured via TOML files. See `torchtitan/config/job_config.py` for all options.

**Extensibility**: Components are pluggable via `TrainSpec` protocol in `protocols/train_spec.py`.
