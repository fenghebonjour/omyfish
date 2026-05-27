import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


def evaluate(config_path: str = "configs/config.yaml", checkpoint: str = "checkpoints/best.pt"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from src.dataset import build_dataloaders
    from src.model import build_model

    _, val_loader, classes = build_dataloaders(config)
    config["model"]["num_classes"] = len(classes)

    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            preds = model(images.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))

    out_dir = Path(config["paths"]["outputs"])
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(classes)
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(max(10, n // 2), max(8, n // 2)))
    sns.heatmap(cm, annot=n <= 30, fmt="d", xticklabels=classes, yticklabels=classes, ax=ax, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix → {cm_path}")
    plt.close()


def gradcam_heatmap(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """
    Grad-CAM heatmap for a single image tensor (1, C, H, W).
    Works for EfficientNet/ResNet backbones; ViT needs a different target layer.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    children = list(model.backbone.children())
    target_layers = [children[-1]]
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

    with GradCAM(model=model, target_layers=target_layers) as cam:
        return cam(input_tensor=image_tensor, targets=targets)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)
