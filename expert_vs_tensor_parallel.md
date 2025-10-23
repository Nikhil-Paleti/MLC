
# Expert Parallelism vs Tensor Parallelism (MoE Cheat Sheet)

## 🧩 Overview

| Aspect | **Tensor Parallel (TP)** | **Expert Parallel (EP)** |
|--------|---------------------------|---------------------------|
| **What it splits** | Splits **individual layers** (matrix/tensor dimensions) across GPUs | Splits **experts (FFNs)** across GPUs |
| **Granularity** | Intra-layer (within same dense layer) | Inter-expert (different experts on different GPUs) |
| **Used in** | Dense transformer layers (QKV, MLP) | Sparse MoE layers |
| **Goal** | Fit large model weights across multiple devices | Scale parameters without scaling compute per token |

---

## ⚙️ How It Works

### 🧠 Tensor Parallel
- A single layer's **weight matrix W** is partitioned across GPUs.
- Each GPU computes partial results → combined via collectives.
- Example (column split):  
  `[W0 | W1 | W2 | W3]` across 4 GPUs  
  `y_i = x @ Wi  →  all_gather(y_i) → concat → y`
- **Communication:** heavy (requires `all_reduce` or `all_gather` after each layer).
- **Pros:** Great for large dense layers.
- **Cons:** Frequent syncs, limited by interconnect bandwidth.

### 🧮 Expert Parallel
- Each GPU owns a subset of **experts** in an MoE layer.
- Router sends tokens → correct experts via **All-to-All**.
- Each GPU processes its own experts → results sent back (another All-to-All).
- Comm pattern:  
  `Router → All-to-All (dispatch) → Experts → All-to-All (combine)`
- **Communication:** dominated by All-to-All per MoE layer.
- **Pros:** Compute scales sub-linearly with model size (sparse activation).
- **Cons:** Routing imbalance, All-to-All latency.

---

## 🔁 Communication Comparison

| Parallelism | Collective Used | Frequency | Data Moved | Dominant Cost |
|--------------|----------------|------------|-------------|----------------|
| **Tensor** | `all_reduce`, `all_gather`, `reduce_scatter` | Every dense layer | Activations / partial outputs | Bandwidth |
| **Expert** | `all_to_all` | Twice per MoE layer | Tokens (routed to experts) | Latency + imbalance |

---

## 🔩 Integration in MoE Frameworks

| Framework | How It Combines Parallelisms |
|------------|------------------------------|
| **Megatron-LM / DeepSpeed-MoE** | Expert + Tensor + Data Parallel (and sometimes Pipeline) |
| **vLLM / FasterMoE / GShard** | Expert parallel for token routing, dynamic load balance |
| **Typical setup:** | `DataParallel × TensorParallel × ExpertParallel` hybrid |

---

## ⚡ Interview Soundbites

- “Tensor parallel splits **weights**, expert parallel splits **experts**.”
- “TP uses **all-reduce**, EP uses **all-to-all**.”
- “All-reduce aggregates partial outputs; All-to-All redistributes tokens.”
- “MoE uses EP to scale model capacity without more compute per token.”
- “EP and TP often coexist: experts are themselves tensor-parallel.”
- “Main EP bottleneck: communication imbalance and All-to-All latency.”
