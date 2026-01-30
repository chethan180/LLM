# LLM Training Usage and Cost Estimates for 1B/3B/8B/70B Models

Calculate the floating point operations for 1B/3B/8B/70B models.

## Technical Details: The Math Behind It
We used two main ways to calculate the computational effort (FLOPs - Floating Point Operations):

### 1. The "FLOPS" Approximation
This is a standard rule of thumb for estimating training cost.
- **Formula:** `FLOPs = 6 * Number_of_Params * Number_of_Tokens`
- **Why 6?** The factor of 6 accounts for the total operations in one training step (Forward + Backward):
  - **Forward Pass:** `2 * N * D` (1 multiply + 1 accumulate per parameter).
  - **Backward Pass:** `4 * N * D` (Calculating gradients for weights and input).
  - **Total:** `2 (Forward) + 4 (Backward) = 6` FLOPs per parameter per token.

### 2. Attention-Aware Formula (More Precise)
For a single micro batch, we calculated:
`FLOPS = (6 * seq_len * num_params) + (12 * num_layers * hidden_size * seq_len^2)`
- This includes the "attention mechanism" calculations which grow quadratically with sequence length.

## Configurations & Results
We ran two types of estimates: one for the total training duration and another for the computational cost of a single training step.

### 1. Training Duration Estimates
Parameters used:
- **Hardware:** 8x NVIDIA H100 GPUs
- **Utilization:** 30% MFU

| Model Size | Training Tokens | Estimated Time |
| :--- | :--- | :--- |
| **1B** Parameters | 20 Billion | ~0.70 days |
| **3B** Parameters | 40 Billion | ~4.17 days |
| **8B** Parameters | 100 Billion | ~27.82 days |
| **70B** Parameters | 240 Billion | ~584.27 days |

### 2. Per-Step Compute Intensity (PFLOPs)
Parameters used:
- **Sequence Length:** 2048
- **Global Batch Size:** 32

| Model Size | PFLOPs per Step |
| :--- | :--- |
| **1B** | **0.39** |
| **3B** | **1.18** |
| **8B** | **3.15** |
| **70B** | **27.53** |

## Hardware Assumptions
- **GPU:** NVIDIA H100
- **Peak Performance:** 832 TFLOPS (Tera-FLOPS) per GPU (FP16).
- **Utilization (MFU):** We assumed we can only use **30% (0.3)** of the theoretical peak speed due to communication overhead and software inefficiencies.
- **System:** 8 GPUs working together.

## TO DO List
We are planning to expand this work with the following items:

- [ ] **Scale to 16 GPUs:** Update calculations for a larger cluster.
- [ ] **Memory & Sharding Analysis:**
  - Calculate max memory usage with **Zero2** vs **Zero3** sharding.
  - Investigate memory requirements for each sharding technique.
- [ ] **Create Configuration Files (YAML):**
  - Create 4 separate config files for **1B, 3B, 8B, and 70B** models.
  - specific open-source details: Sequence length, Number of Experts (MoE), Attention Heads.
  - For each config:
    1. Calculate total FLOPs.
    2. Calculate total memory required.
- [ ] **Hardware Comparison (H100 vs Blackwell):**
  - Compare Total VRAM per machine.
  - Compare Max TFLOPS achievable for each datatype.
- [ ] **Detailed Resource Planning:**
  - **Sequence Length & Batch Size:** Analyze impact on memory.
  - **Memory per GPU:**
    - Estimate specific holdings (e.g., 1B model → ~16GB data).
    - Estimate for 70B model (scaled).
  - **Trade-offs:** Gradient Accumulation vs. Activation Checkpointing (Memory vs. Compute).
  - **Starting Loss:** Estimate starting loss for 1B tokens (1 epoch).
- [ ] **Final Output Estimations:**
  - Map Config → Machine (FLOPs per datatype, Optimization strategies).
  - Total hours required to train.
  - Max memory peak observed per machine.


## References
- [Hugging Face Nanotron Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=broadcast) 