import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from services.fish_ai.model.classifier import build_model
from services.fish_ai.predictors.base import BaseFishPredictor
from services.fish_ai.training.transforms import get_val_transforms

UNCERTAIN_THRESHOLD = 0.30


def _normalize(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


class FishPredictor(BaseFishPredictor):
    def __init__(
        self,
        checkpoint_path: str,
        metadata_path: str = "data/metadata/fish_info.json",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = ckpt["config"]

        classes_path = Path(checkpoint_path).parent / "classes.json"
        self.classes = json.loads(classes_path.read_text())
        config["model"]["num_classes"] = len(self.classes)

        self.model = build_model(config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.transform = get_val_transforms(config["data"]["image_size"])

        fish_list = json.loads(Path(metadata_path).read_text())
        self.metadata = {_normalize(entry["species"]): entry for entry in fish_list}

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        arr = np.array(image.convert("RGB"))
        tensor = self.transform(image=arr)["image"].unsqueeze(0).to(self.device)

        probs = F.softmax(self.model(tensor), dim=1)[0]
        top_probs, top_idx = probs.topk(min(top_k, len(self.classes)))

        predictions = []
        for prob, idx in zip(top_probs.tolist(), top_idx.tolist()):
            name = self.classes[idx]
            predictions.append({
                "species": name,
                "confidence": round(prob, 4),
                "metadata": self.metadata.get(_normalize(name), {}),
            })

        uncertain = predictions[0]["confidence"] < UNCERTAIN_THRESHOLD
        return {
            "predictions": predictions,
            "uncertain": uncertain,
            "message": "Low confidence — species may be outside training distribution." if uncertain else None,
        }

    def export_onnx(self, output_path: str = "checkpoints/model.onnx", image_size: int = 300):
        dummy = torch.randn(1, 3, image_size, image_size).to(self.device)
        torch.onnx.export(
            self.model, dummy, output_path,
            input_names=["image"], output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=18,
        )
        print(f"ONNX model exported → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    predictor = FishPredictor(args.checkpoint)
    result = predictor.predict(Image.open(args.image), top_k=args.top_k)

    for i, p in enumerate(result["predictions"], 1):
        print(f"{i}. {p['species']:<30s} {p['confidence'] * 100:.1f}%")
    if result["uncertain"]:
        print(f"\nWarning: {result['message']}")
