"""
Zero-shot fish identifier using CLIP. No training or dataset required.
Uses fish species names from fish_info.json as text prompts.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

UNCERTAIN_THRESHOLD = 0.08  # with 30 classes, random = ~3%; real hits cluster above 10%
MODEL_ID = "openai/clip-vit-base-patch32"


def _normalize(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


class CLIPFishPredictor:
    def __init__(self, metadata_path: str = "data/metadata/fish_info.json"):
        self.model = CLIPModel.from_pretrained(MODEL_ID)
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)
        self.model.eval()

        fish_list = json.loads(Path(metadata_path).read_text())
        self.metadata = {_normalize(f["species"]): f for f in fish_list}
        self.species = [f["species"] for f in fish_list]
        # Descriptive prompts outperform bare class names for CLIP
        self.prompts = [
            f"a photo of a {s.replace('_', ' ')} fish" for s in self.species
        ]

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        inputs = self.processor(
            text=self.prompts,
            images=image.convert("RGB"),
            return_tensors="pt",
            padding=True,
        )
        probs = self.model(**inputs).logits_per_image.softmax(dim=1)[0]
        top_probs, top_idx = probs.topk(min(top_k, len(self.species)))

        predictions = []
        for prob, idx in zip(top_probs.tolist(), top_idx.tolist()):
            name = self.species[idx]
            predictions.append({
                "species": name.replace("_", " ").title(),
                "confidence": round(prob, 4),
                "metadata": self.metadata.get(_normalize(name), {}),
            })

        uncertain = predictions[0]["confidence"] < UNCERTAIN_THRESHOLD
        return {
            "predictions": predictions,
            "uncertain": uncertain,
            "message": "Low confidence — this may not be a fish, or the species isn't in our database." if uncertain else None,
        }
