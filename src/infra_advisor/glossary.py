"""Single source of truth for plain-English term definitions.

Two presentations of the same domain vocabulary live here so they no longer drift
across files:

- ``GLOSSARY``: verbose definitions for the full report appendix (report.py).
- ``SHORT_GLOSSARY``: concise inline definitions for follow-up answers (followup.py).

Keep shared facts (prices, ratios) consistent between the two when editing.
"""

# Verbose definitions — rendered in the full-report glossary appendix.
GLOSSARY = {
    "GPU": "Graphics Processing Unit — specialized chip that runs AI workloads far faster than a regular CPU.",
    "VRAM": "Video RAM — memory on the GPU chip. Models must fit entirely in VRAM to run. More VRAM = larger models.",
    "Parameters (B)": "The number of learned values in a model. A '7B' model has 7 billion parameters. Larger = smarter but slower and more expensive.",
    "Context Window": "Maximum text length the model can read at once. 128K tokens ≈ ~100,000 words or ~300 pages.",
    "Token": "A unit of text (~0.75 words on average). Pricing and speed are measured in tokens.",
    "FLOPs": "Floating Point Operations — a measure of compute work. More FLOPs = more compute time and cost.",
    "ExaFLOP": "1 quintillion (10¹⁸) FLOPs. Used to express large training runs.",
    "MFU (Model FLOPs Utilization)": "How efficiently the GPU is being used. 50% MFU means the GPU is running at half its theoretical peak — typical for well-tuned training.",
    "GPU-hours": "Total compute time across all GPUs combined. 8 GPUs running for 10 hours = 80 GPU-hours.",
    "Pre-training": "Training a model from scratch on massive datasets (trillions of tokens). Extremely expensive. Done by labs like Meta and Google.",
    "Continual Pre-training (CPT)": "Continuing pre-training on a new domain-specific corpus to shift the model's knowledge. Requires billions of tokens to be meaningful.",
    "SFT (Supervised Fine-Tuning)": "Teaching a pre-trained model to follow instructions or specialize in a task using curated examples. Far cheaper than pre-training.",
    "RL / RLHF": "Reinforcement Learning from Human Feedback — used to align model behavior. More expensive than SFT due to rollout overhead.",
    "RAG (Retrieval-Augmented Generation)": "Architecture where the model retrieves relevant documents at query time instead of memorizing everything. Reduces hallucination and VRAM needs.",
    "Inference": "Running the model to generate responses (as opposed to training). What your users experience in production.",
    "Throughput (tokens/sec)": "How many tokens the model generates per second. Higher = faster responses and lower cost per query.",
    "TCO (Total Cost of Ownership)": "All costs over a time period: hardware purchase, power, cooling, labor, and depreciation.",
    "CapEx": "Capital Expenditure — one-time hardware purchase cost.",
    "OpEx": "Operational Expenditure — ongoing monthly costs (power, cooling, staff, etc.).",
    "Break-even": "The month at which cumulative on-prem costs drop below cumulative cloud costs.",
    "Spot Instances": "Spare cloud capacity sold at 35–65% discount. Can be interrupted with short notice. Good for training, risky for production inference.",
    "On-demand Instances": "Standard cloud pricing — available immediately, never interrupted, no commitment.",
    "FTE": "Full-Time Equivalent — one full-time employee. 0.5 FTE = half a person's time.",
    "PUE (Power Usage Effectiveness)": "Data center efficiency ratio. PUE of 1.4 means 40% extra power is used for cooling beyond what the GPUs consume.",
    "Data Parallel (DDP)": "Replicating the full model on each GPU and averaging gradients. The simplest multi-GPU training when the model fits on one GPU.",
    "Tensor Parallelism (TP)": "Splitting each layer's math across multiple GPUs in the same machine. Needs fast NVLink interconnect. Used to fit or serve a model too big for one GPU.",
    "Pipeline Parallelism (PP)": "Splitting a model's layers into stages across GPUs/nodes; each stage handles part of the network. Used when one replica spans multiple machines.",
    "FSDP / ZeRO-3": "Sharding a model's parameters, gradients, and optimizer states across GPUs so a model too large for one GPU can still be trained (PyTorch FSDP / DeepSpeed ZeRO-3).",
}

# Concise definitions — rendered inline at the bottom of each follow-up answer.
SHORT_GLOSSARY = {
    "token": "A unit of text (~0.75 words). All AI pricing and speed is measured in tokens.",
    "GPU": "Graphics Processing Unit — specialized chip that runs AI math far faster than a CPU.",
    "VRAM": "GPU memory. The model must fit entirely in VRAM to run.",
    "inference": "Running the model to generate responses. Your ongoing monthly production cost.",
    "throughput": "Tokens generated per second. Higher = faster responses and lower cost per query.",
    "SFT": "Supervised Fine-Tuning — training on curated (input → ideal output) pairs to specialize a model.",
    "LoRA / QLoRA": "Memory-efficient fine-tuning. Trains small adapter layers only — 4–8× cheaper than full fine-tuning.",
    "RLHF": "Reinforcement Learning from Human Feedback — uses human preference labels to align model behavior.",
    "DPO": "Direct Preference Optimization — simpler RLHF alternative. 3–8× cheaper, comparable quality for most tasks.",
    "TCO": "Total Cost of Ownership — hardware purchase + power + cooling + staff over a time period.",
    "CapEx": "Capital Expenditure — one-time hardware purchase cost.",
    "OpEx": "Operational Expenditure — ongoing monthly costs (power, cooling, maintenance, staff).",
    "break-even": "Month when cumulative on-prem cost drops below cumulative cloud cost.",
    "spot instances": "Spare cloud capacity at 35–65% discount. Can be interrupted with 2 min notice. Good for training; risky for production.",
    "vLLM": "Open-source serving engine. Continuous batching lets one GPU serve many concurrent users efficiently.",
    "RAG": "Retrieval-Augmented Generation — fetch relevant documents at query time rather than memorizing them.",
    "FTE": "Full-Time Equivalent — one full-time employee. 0.5 FTE = half a person's time.",
    "quantization": "Compressing model weights (16-bit → 8-bit or 4-bit) to reduce VRAM and increase speed, with <1% quality loss.",
    "H100": "NVIDIA flagship AI GPU. 80GB VRAM, ~$30K/unit. Best for large model training and serving.",
    "A100": "Previous-gen NVIDIA flagship. 40–80GB VRAM, ~$10–15K. Competitive for inference.",
    "L40S": "NVIDIA inference-focused GPU. 48GB VRAM, ~$8K. Best price/performance for inference workloads.",
    "MFU": "Model FLOPs Utilization — how efficiently the GPU is used. 50% MFU means running at half theoretical peak.",
    "ExaFLOP": "10¹⁸ floating-point operations. Used to express large training run sizes.",
    "tensor parallel": "Split each layer across GPUs in one machine (needs NVLink). Serves a model too big for one GPU.",
    "pipeline parallel": "Split the model's layers into stages across GPUs/nodes. Used when one replica spans machines.",
    "FSDP / ZeRO-3": "Shard parameters, gradients, and optimizer states across GPUs to train a model too large for one GPU.",
}
