import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FishClassifier(nn.Module):
    def __init__(self, architecture: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(architecture, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalized embedding — use for similarity search."""
        return F.normalize(self.backbone(x), dim=1)


def build_model(config: dict) -> FishClassifier:
    return FishClassifier(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    )
