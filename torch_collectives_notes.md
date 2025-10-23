
# PyTorch Collective Communication – Interview Notes

## 🧩 Core Collectives (Concepts)

| Collective | Purpose | Equivalent Composition | Communication Cost (bytes per process) |
|-------------|----------|------------------------|----------------------------------------|
| **All-Reduce** | Combine tensors (sum/avg/max) from all ranks → everyone gets result | Reduce → Broadcast | `O((p-1)/p * N)` ≈ `2N` per process on ring |
| **Reduce** | Combine tensors and send result to one destination rank | N/A | `O((p-1)/p * N)` (only root receives result) |
| **Broadcast** | Send tensor from one source rank to all ranks | N/A | `O((p-1)/p * N)` |
| **All-Gather** | Each rank gathers tensors from all others (concatenation) | Gather → Broadcast | `O((p-1) * N)` |
| **Reduce-Scatter** | Reduce (sum/avg) across ranks then scatter the result chunks | All-Reduce + Scatter | `O((p-1)/p * N)` |
| **All-to-All** | Each rank sends distinct chunk to every other rank | p simultaneous sends/recvs | `O((p-1)/p * N)` |
| **Barrier** | Synchronize all ranks (no data transfer) | Broadcast of zero bytes | Negligible |

---

## 🔍 Intuition & Decomposition

| Operation | Conceptual Steps |
|------------|-----------------|
| **All-Reduce** | 1️⃣ Each rank reduces (sums) partial results <br>2️⃣ Result is broadcast back to all ranks |
| **Reduce-Scatter** | 1️⃣ Perform an all-reduce <br>2️⃣ Split the result into equal chunks (each rank keeps its own) |
| **All-Gather** | 1️⃣ Each rank sends its chunk to everyone <br>2️⃣ Equivalent to *Gather + Broadcast* |
| **All-to-All** | 1️⃣ Each rank partitions tensor into p parts <br>2️⃣ Exchanges them with all others (full matrix shuffle) |
| **Broadcast** | One-to-many replication (rank 0 → all) |
| **Reduce** | Many-to-one aggregation (all → rank 0) |

---

## 🚀 Communication Patterns (for GPUs via NCCL)

- **Ring algorithm** – bandwidth-optimal for large tensors:  
  Time ≈ `2 * (p-1)/p * (N / BW)`
- **Tree algorithm** – low-latency for small tensors:  
  Time ≈ `log(p)` × per-message latency

---

## 🧠 Usage Cheat Snippets

```python
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
dist.reduce(tensor, dst=0)
dist.broadcast(tensor, src=0)

bufs = [torch.empty_like(tensor) for _ in range(world)]
dist.all_gather(bufs, tensor)

out = torch.empty_like(tensor)
dist.reduce_scatter(out, [tensor.clone() for _ in range(world)])

dist.all_to_all_single(out, tensor)
dist.barrier()
```

---

## 📊 Quick Comparison Summary

| Operation | Everyone gets full result? | Uses reduction? | Notes |
|------------|-----------------------------|------------------|-------|
| **Broadcast** | ✅ | ❌ | From one → all |
| **Reduce** | ❌ | ✅ | From all → one |
| **All-Reduce** | ✅ | ✅ | Combines + shares |
| **All-Gather** | ✅ | ❌ | Concatenates all |
| **Reduce-Scatter** | ❌ | ✅ | Memory-efficient all-reduce |
| **All-to-All** | ❌ | ❌ | Arbitrary shuffling |
| **Barrier** | ❌ | ❌ | Sync only |

---

## 💡 Interview Nuggets

- `Reduce-Scatter + All-Gather = All-Reduce`
- DDP internally implements **All-Reduce** as **Reduce-Scatter + All-Gather**
- **All-to-All** → used in MoE routing
- Use `async_op=True` for overlap
- Latency dominates small tensors; bandwidth dominates large tensors
