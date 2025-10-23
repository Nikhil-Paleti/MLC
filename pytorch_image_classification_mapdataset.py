
import argparse, os, time, random
from typing import List, Tuple, Optional, Callable, Dict
from dataclasses import dataclass

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

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

# ===============================
# Simple Transform Helpers (pure-PIL/NumPy/Torch)
# ===============================
@dataclass
class Compose:
    transforms: List[Callable]
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class Resize:
    def __init__(self, size: int):
        self.size = size
    def __call__(self, img: Image.Image):
        return img.resize((self.size, self.size), Image.BILINEAR)

class CenterCrop:
    def __init__(self, size: int):
        self.size = size
    def __call__(self, img: Image.Image):
        w, h = img.size
        th, tw = self.size, self.size
        i = max(0, (h - th) // 2)
        j = max(0, (w - tw) // 2)
        return img.crop((j, i, j + tw, i + th))

class RandomResizedCrop:
    def __init__(self, size: int, scale=(0.8, 1.0)):
        self.size = size
        self.scale = scale
    def __call__(self, img: Image.Image):
        w, h = img.size
        area = w * h
        for _ in range(10):
            target = random.uniform(*self.scale) * area
            aspect = random.uniform(3/4, 4/3)
            nw = int(round((target * aspect) ** 0.5))
            nh = int(round((target / aspect) ** 0.5))
            if nw <= w and nh <= h and nw > 0 and nh > 0:
                j = random.randint(0, w - nw)
                i = random.randint(0, h - nh)
                img = img.crop((j, i, j + nw, i + nh))
                return img.resize((self.size, self.size), Image.BILINEAR)
        return CenterCrop(self.size)(Resize(self.size)(img))

class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p
    def __call__(self, img: Image.Image):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class ToTensor:
    def __call__(self, img: Image.Image):
        arr = np.array(img, dtype=np.float32, copy=False)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        arr = arr / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return t

class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = torch.tensor(mean)[:, None, None]
        self.std = torch.tensor(std)[:, None, None]
    def __call__(self, t: torch.Tensor):
        return (t - self.mean) / (self.std + 1e-12)

# ===============================
# Map-style Image Dataset (ImageFolder-like)
# ===============================
class ImageFolderLite(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None, extensions={".jpg", ".jpeg", ".png", ".bmp"}):
        self.root = root
        self.transform = transform
        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(classes)}
        self.samples: List[Tuple[str, int]] = []
        for c in classes:
            d = os.path.join(root, c)
            for fname in os.listdir(d):
                ext = os.path.splitext(fname)[1].lower()
                if ext in extensions:
                    self.samples.append((os.path.join(d, fname), self.class_to_idx[c]))
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root}. Expected structure: root/class_x/xxx.jpg")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

# ===============================
# Model
# ===============================
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.head(x)

def maybe_resnet18(num_classes: int):
    try:
        import torchvision.models as models
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    except Exception:
        return SmallCNN(num_classes)

# ===============================
# DataLoader builders
# ===============================
def build_loaders(args, device):
    normalize = Normalize()
    train_tfms = Compose([RandomResizedCrop(args.image_size), RandomHorizontalFlip(0.5), ToTensor(), normalize])
    val_tfms   = Compose([Resize(args.image_size), CenterCrop(args.image_size), ToTensor(), normalize])

    train_ds = ImageFolderLite(os.path.join(args.data_root, "train"), transform=train_tfms)
    val_ds   = ImageFolderLite(os.path.join(args.data_root, "val"),   transform=val_tfms)

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if dist.is_initialized() else None
    val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=False) if dist.is_initialized() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory and device.type == "cuda"),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory and device.type == "cuda"),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        drop_last=False,
    )
    return train_loader, val_loader, train_sampler, len(train_ds.class_to_idx)

# ===============================
# Train / Eval
# ===============================

# def train_one_epoch(model, loader, optimizer, device, epoch, sampler=None):
#     model.train()
#     if sampler is not None:
#         sampler.set_epoch(epoch)

#     total = correct = 0
#     data_time = step_time = 0.0
#     end = time.perf_counter()

#     for step, (x, y) in enumerate(loader):
#         data_time += time.perf_counter() - end

#         x = x.to(device, non_blocking=True)
#         y = y.to(device, non_blocking=True)

#         optimizer.zero_grad(set_to_none=True)
#         logits = model(x)                                # no autocast
#         loss = nn.functional.cross_entropy(logits, y)

#         loss.backward()                                  # no scaler
#         optimizer.step()

#         pred = logits.argmax(dim=1)
#         total += y.size(0)
#         correct += (pred == y).sum().item()

#         step_time += time.perf_counter() - end
#         end = time.perf_counter()

#     acc = correct / max(1, total)
#     return acc, data_time, step_time

def train_one_epoch(model, loader, optimizer, scaler, device, epoch, sampler=None):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    total, correct = 0, 0
    data_time, step_time = 0.0, 0.0
    end = time.perf_counter()

    for step, (x, y) in enumerate(loader):
        data_time += time.perf_counter() - end
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pred = logits.argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()

        step_time += time.perf_counter() - end
        end = time.perf_counter()

    acc = correct / max(1, total)
    return acc, data_time, step_time

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return correct / max(1, total)

# ===============================
# Main
# ===============================
def main():
    parser = argparse.ArgumentParser(description="Image classification with Map-style Dataset + DataLoader best practices")
    parser.add_argument("--data_root", type=str, required=True, help="Root dir with train/ and val/ subfolders (ImageFolder style)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "smallcnn", "resnet18"])
    args = parser.parse_args()

    # DDP Init if launched by torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, train_sampler, num_classes = build_loaders(args, device)

    if args.model == "smallcnn":
        model = SmallCNN(num_classes)
    elif args.model == "resnet18":
        try:
            import torchvision.models as models
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        except Exception:
            print("torchvision not available, falling back to SmallCNN")
            model = SmallCNN(num_classes)
    else:
        model = maybe_resnet18(num_classes)

    model = model.to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], output_device=int(os.environ["LOCAL_RANK"]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(args.epochs):
        train_acc, data_t, step_t = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, train_sampler)
        val_acc = evaluate(model, val_loader, device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[Epoch {epoch:02d}] train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
                  f"| data_time={data_t:.2f}s step_time={step_t:.2f}s "
                  f"| nw={args.num_workers} prefetch={args.prefetch_factor} pin={args.pin_memory}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
