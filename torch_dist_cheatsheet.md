# PyTorch Distributed (torch.distributed) Cheat Sheet

## 1. Initialization

### Single node (multi-process, 1 GPU per process)

```bash
torchrun --standalone --nproc_per_node=8 train.py
```

```python
import os, torch, torch.distributed as dist

def setup():
    dist.init_process_group(
        backend="nccl",        # nccl (GPU), gloo (CPU), mpi (optional)
        init_method="env://",  # torchrun sets env vars
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()
```

### Multi-node (N nodes × P GPUs)

```bash
# On node 0
MASTER_ADDR=node0 MASTER_PORT=29500 torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=node0:29500 train.py

# On node 1
MASTER_ADDR=node0 MASTER_PORT=29500 torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=node0:29500 train.py
```

### Backends

- **NCCL** – best for GPU tensors (Linux)
- **Gloo** – CPU tensors or debugging
- **MPI** – if PyTorch built with MPI support

### Environment variables set by torchrun

```
RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
```

```python
rank = dist.get_rank()
world = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
```

---

## 2. Process Groups

Used to form subsets of ranks (for tensor/pipeline parallelism).

```python
# must be called in same order on all ranks
tp_group = dist.new_group(ranks=[0,1,2,3])
pp_group = dist.new_group(ranks=[0,4])

# collective within subgroup
tensor = torch.ones(1).cuda()
dist.all_reduce(tensor, group=tp_group)
```

> ⚠️ All processes must call new\_group in **identical order**, even if they are not members.

---

## 3. Collective Communication Ops

### Common Collectives

```python
t = torch.ones(4, device="cuda")

# 1. All-reduce (sum/avg/max over all ranks)
dist.all_reduce(t, op=dist.ReduceOp.SUM)

# 2. Broadcast (rank 0 -> all)
dist.broadcast(t, src=0)

# 3. Reduce (to single dst rank)
dist.reduce(t, dst=0, op=dist.ReduceOp.SUM)

# 4. All-gather (same-shaped tensors from all ranks)
bufs = [torch.empty_like(t) for _ in range(dist.get_world_size())]
dist.all_gather(bufs, t)

# 5. Reduce-scatter
dist.reduce_scatter(t, [torch.ones_like(t) for _ in range(world)])

# 6. All-to-all
dist.all_to_all_single(out_tensor, in_tensor)

# 7. Barrier (sync)
dist.barrier()
```

### Point-to-Point (P2P)

```python
if rank == 0:
    dist.send(t, dst=1)
elif rank == 1:
    dist.recv(t, src=0)
```

---

## 4. Async Collectives

```python
work = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
# do compute here
work.wait()
```

---

## 5. Minimal Example

```python
# train.py
import os, torch, torch.distributed as dist

def setup():
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def train_step(x):
    loss = x.sum()
    dist.all_reduce(loss)
    return loss / dist.get_world_size()

def main():
    setup()
    x = torch.ones(2, device="cuda")
    loss = train_step(x)
    if dist.get_rank() == 0:
        print("loss:", loss.item())
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

---

## 6. Gotchas

- NCCL only supports GPU tensors.
- All ranks must enter the same collectives in the same order.
- Tensor shapes/dtypes/devices must match across ranks.
- Environment variables must be consistent across nodes.
- `async_op=True` lets you overlap compute and comm.
- DDP uses these primitives internally for gradient synchronization.

---

## 7. Useful Commands

```bash
# Single-node 8 GPUs
torchrun --standalone --nproc_per_node=8 train.py

# 2 nodes × 8 GPUs
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=node0:29500 train.py
```

