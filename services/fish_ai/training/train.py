import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from services.fish_ai.training.dataset import build_dataloaders
from services.fish_ai.model.classifier import build_model
from services.fish_ai.training.utils import seed_everything


def _top5(logits: torch.Tensor, labels: torch.Tensor) -> float:
    k = min(5, logits.size(1))
    _, top = logits.topk(k, dim=1)
    return top.eq(labels.view(-1, 1).expand_as(top)).any(1).float().mean().item()


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience, self.best, self.count = patience, float("inf"), 0

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best - 1e-4:
            self.best, self.count = val_loss, 0
        else:
            self.count += 1
        return self.count >= self.patience


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = top5_sum = n = 0

    for images, labels in tqdm(loader, desc="train" if training else "val  ", leave=False):
        images, labels = images.to(device), labels.to(device)

        if training:
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, labels)

        total_loss += loss.item()
        correct += (logits.detach().argmax(1) == labels).sum().item()
        top5_sum += _top5(logits.detach(), labels) * labels.size(0)
        n += labels.size(0)

    return total_loss / len(loader), correct / n, top5_sum / n


def train(config_path: str = "configs/training.yaml", resume: str = None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed_everything(42)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    train_loader, val_loader, classes = build_dataloaders(config)
    config["model"]["num_classes"] = len(classes)
    print(f"Classes: {len(classes)}")

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    stopper = EarlyStopping(config["training"]["early_stopping_patience"])
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    use_wandb = config["logging"]["use_wandb"]
    if use_wandb:
        import wandb
        wandb.init(project=config["logging"]["project"], config=config)

    ckpt_dir = Path(config["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "classes.json").write_text(json.dumps(classes))

    best_acc = 0.0
    start_epoch = 1

    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt["best_acc"]
        print(f"Resumed from epoch {ckpt['epoch']} (best_acc={best_acc:.4f})")

    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc, _ = _run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        vl_loss, vl_acc, vl_top5 = _run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"[{epoch:03d}] train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"val loss={vl_loss:.4f} acc={vl_acc:.3f} top5={vl_top5:.3f} | "
            f"{time.time() - t0:.0f}s"
        )

        if use_wandb:
            import wandb
            wandb.log({"train/loss": tr_loss, "train/acc": tr_acc,
                       "val/loss": vl_loss, "val/acc": vl_acc, "val/top5": vl_top5})

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "config": config, "val_acc": vl_acc},
                ckpt_dir / "best.pt",
            )
            print(f"  ✓ saved best (val_acc={vl_acc:.4f})")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "best_acc": best_acc,
                "config": config,
                "val_acc": vl_acc,
            },
            ckpt_dir / "last.pt",
        )

        if stopper(vl_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    if use_wandb:
        import wandb
        wandb.finish()

    print(f"\nBest val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--resume", nargs="?", const="checkpoints/last.pt", default=None,
                        help="Resume from checkpoint (default: checkpoints/last.pt)")
    args = parser.parse_args()
    train(args.config, resume=args.resume)
