
import argparse, os, time, math, random
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, IterableDataset, DataLoader, DistributedSampler

# ===============================
# Utilities
# ===============================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id: int):
    info = torch.utils.data.get_worker_info()
    base_seed = info.seed
    np.random.seed(base_seed % (2**32 - 1))
    random.seed(base_seed)

def collate_pad(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.int32)
    D = xs[0].shape[1]
    max_len = int(lengths.max())
    out = xs[0].new_zeros(len(xs), max_len, D)
    for i, x in enumerate(xs):
        out[i, :x.shape[0]] = x
    return out, torch.tensor(ys, dtype=torch.long), lengths

class DevicePrefetcher:
    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None
        self.iter = None
        self.next_batch = None

    def __iter__(self):
        self.iter = iter(self.loader)
        if self.stream is None:
            return self
        self._preload()
        return self

    def __next__(self):
        if self.stream is None:
            return next(self.iter)
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_batch is None:
            raise StopIteration
        batch = self.next_batch
        self._preload()
        return batch

    def _to_device(self, obj):
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._to_device(o) for o in obj)
        if isinstance(obj, dict):
            return {k: self._to_device(v) for k, v in obj.items()}
        return obj

    def _preload(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_batch = self._to_device(batch)
        else:
            self.next_batch = batch

# ===============================
# Datasets
# ===============================
class ToyMapDataset(Dataset):
    def __init__(self, n_samples: int, max_len: int = 64, dim: int = 32):
        self.n = n_samples
        self.max_len = max_len
        self.dim = dim

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        L = np.random.randint(8, self.max_len + 1)
        x = torch.from_numpy(np.random.randn(L, self.dim).astype(np.float32))
        y = int(x.mean() > 0)
        return x, y

class ToyIterableDataset(IterableDataset):
    def __init__(self, n_samples: int, max_len: int = 64, dim: int = 32):
        self.n = n_samples
        self.max_len = max_len
        self.dim = dim

    def __iter__(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        start = math.floor(self.n * rank / world)
        end = math.floor(self.n * (rank + 1) / world)
        for _ in range(start, end):
            L = np.random.randint(8, self.max_len + 1)
            x = torch.from_numpy(np.random.randn(L, self.dim).astype(np.float32))
            y = int(x.mean() > 0)
            yield x, y

# ===============================
# Model
# ===============================
class TinyRNNClassifier(nn.Module):
    def __init__(self, dim: int = 32, hidden: int = 64, num_classes: int = 2):
        super().__init__()
        self.rnn = nn.GRU(dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        logits = self.head(h[-1])
        return logits

# ===============================
# DataLoader builders
# ===============================
def build_map_loader(args, is_train=True):
    ds = ToyMapDataset(n_samples=args.n_samples, max_len=args.max_len, dim=args.dim)
    sampler = None
    if dist.is_initialized():
        sampler = DistributedSampler(ds, shuffle=is_train, drop_last=is_train)
    dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(sampler is None and is_train),
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and torch.cuda.is_available(),
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            persistent_workers=(args.num_workers > 0),
            collate_fn=collate_pad,
            worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
            drop_last=is_train,
        )
    return dl, sampler

def build_iter_loader(args):
    ds = ToyIterableDataset(n_samples=args.n_samples, max_len=args.max_len, dim=args.dim)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_pad,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )
    return dl

# ===============================
# Train / Eval
# ===============================
def train_one_epoch(model, loader, optimizer, scaler, device, epoch, sampler=None, use_prefetch=True):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    it = DevicePrefetcher(loader, device) if (use_prefetch and device.type == "cuda") else loader
    total, correct, data_time, step_time = 0, 0, 0.0, 0.0
    end = time.perf_counter()

    for step, batch in enumerate(it):
        data_time += time.perf_counter() - end

        if isinstance(batch, (list, tuple)):
            x, y, lengths = batch
        else:
            raise ValueError("Unexpected batch structure")

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            logits = model(x.to(device, non_blocking=True), lengths.to(device, non_blocking=True))
            loss = nn.functional.cross_entropy(logits, y.to(device, non_blocking=True))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pred = logits.argmax(dim=1)
        total += y.size(0)
        correct += (pred.cpu() == y).sum().item()

        step_time += time.perf_counter() - end
        end = time.perf_counter()

    acc = correct / max(1, total)
    return acc, data_time, step_time

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    for x, y, lengths in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        logits = model(x, lengths)
        pred = logits.argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return correct / max(1, total)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--use_iterable", action="store_true", help="Use IterableDataset instead of MapDataset")
    parser.add_argument("--n_samples", type=int, default=8192)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # DDP init if launched by torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build data loaders
    if args.use_iterable:
        train_loader = build_iter_loader(args)
        val_loader = build_iter_loader(args)
        sampler = None
    else:
        train_loader, sampler = build_map_loader(args, is_train=True)
        val_loader, _ = build_map_loader(args, is_train=False)

    model = TinyRNNClassifier(dim=args.dim).to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], output_device=int(os.environ["LOCAL_RANK"]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(args.epochs):
        acc, data_t, step_t = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, sampler)
        val_acc = evaluate(model, val_loader, device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[Epoch {epoch:02d}] train_acc={acc:.3f} val_acc={val_acc:.3f} "
                  f"| data_time={data_t:.2f}s step_time={step_t:.2f}s "
                  f"| nw={args.num_workers} prefetch={args.prefetch_factor} pin={args.pin_memory}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
