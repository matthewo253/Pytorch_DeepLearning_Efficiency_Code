# PyTorch Deep Learning Efficiency & Optimization Lab

A collection of PyTorch implementation experiments exploring training stability, parameter-efficient fine-tuning (PEFT), activation checkpointing, and memory optimization techniques across MLP and CNN architectures.

## Repository Structure

## Experiments Overview

### 1. Batch Normalization Ablation (`scripts/Assignment1_part0.py`)
* Compares standard MLPs and CNNs against Batch Normalization variants (`BatchNorm1d` and `BatchNorm2d`) on MNIST[cite: 6].
* Tracks convergence speed and test accuracy over 15 epochs using SGD[cite: 6].

### 2. Low-Rank Adaptation: LoRA & LoRA-FA (`scripts/Assignment1_part1.py`)
* Implements custom `LoRALinear` layers from scratch with frozen base weights and low-rank decomposition matrices ($A$ and $B$)[cite: 7].
* Compares full fine-tuning against **LoRA** and **LoRA-FA** (Frozen-A) across ranks $r=8$ and $r=160$, measuring trainable parameter counts, per-epoch training delay, and test accuracy[cite: 7].

### 3. Activation Checkpointing (`scripts/Assignment1_part2.py`)
* Uses `torch.utils.checkpoint.checkpoint` to trade compute for reduced peak GPU memory allocation[cite: 8].
* Evaluates memory reduction and training overhead on both an `MLP_BN` and a multi-block `CNN_BN` (testing checkpointing on the first block vs. both convolutional blocks)[cite: 8].

### 4. Gradient Accumulation & OOM Mitigation (`scripts/Assignment1_part3.py`)
* Intentionally triggers a `CUDA out of memory` error using an oversized batch size (`2048`) on CIFAR-10 (`64x64` inputs)[cite: 9].
* Resolves the memory bottleneck using mini-batch training (`batch_size=256`) paired with **Gradient Accumulation** (`accumulation_steps=8`) to replicate an effective batch size of `2048` within hardware memory constraints[cite: 9].

## Installation & Quick Start

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/matthewo253/pytorch-dl-efficiency-lab.git](https://github.com/matthewo253/pytorch-dl-efficiency-lab.git)
   cd pytorch-dl-efficiency-lab
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run any experiment:**
   *(Datasets such as MNIST and CIFAR-10 download automatically into `./data` on the first run[cite: 6, 7, 8, 9]).*
   ```bash
   python scripts/Assignment1_part0.py
   python scripts/Assignment1_part1.py
   python scripts/Assignment1_part2.py
   python scripts/Assignment1_part3.py
   ```
