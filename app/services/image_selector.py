from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from app.repositories.embeddings import EmbeddingsRepository
from app.core.config import settings


@dataclass
class SelectionResult:
    kept: List[str]
    removed: List[str]


class ImageSelectorService:
    def __init__(self, db_repo: Optional[EmbeddingsRepository] = None, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Load feature extractor (ResNet50 without classifier)
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = torch.nn.Sequential(*list(base.children())[:-1]).to(self.device)
        self.feature_extractor.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.repo = db_repo or EmbeddingsRepository(settings.DB_PATH)

        # Lazy-load aesthetics predictor to avoid dependency unless needed
        self._predictor = None
        self._processor = None

    def _ensure_aesthetics(self):
        if self._predictor is None or self._processor is None:
            from transformers import CLIPProcessor
            from aesthetics_predictor import AestheticsPredictorV1

            model_id = "shunk031/aesthetics-predictor-v1-vit-large-patch14"
            self._predictor = AestheticsPredictorV1.from_pretrained(model_id).to(self.device)
            self._processor = CLIPProcessor.from_pretrained(model_id)

    def embed_image(self, image_path: Path) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.feature_extractor(tensor).squeeze().detach().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(emb)
        return emb / max(norm, 1e-8)

    def add_image(self, image_path: Path) -> None:
        emb = self.embed_image(image_path)
        self.repo.upsert(str(image_path), emb.tobytes())

    def predict_aesthetic(self, image_path: Path) -> float:
        self._ensure_aesthetics()
        img = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._predictor(**inputs)
        return float(outputs.logits[0].item())

    def find_similar(self, query_image: Path, threshold: float) -> List[str]:
        q = self.embed_image(query_image)
        entries = self.repo.list_all()
        similar = []
        for path, emb_blob in entries:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            sim = float(np.dot(q, emb))
            if sim >= threshold:
                similar.append(path)
        return similar

    def choose_best(self, input_dir: Path, output_dir: Path, similarity: float = 0.87, use_aesthetics: bool = True) -> SelectionResult:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Index all images
        for file in os.listdir(input_dir):
            fp = input_dir / file
            if fp.is_file():
                try:
                    self.add_image(fp)
                except Exception:
                    continue

        kept: List[str] = []
        removed: List[str] = []

        i = 0
        for file in os.listdir(input_dir):
            fp = input_dir / file
            if not fp.is_file():
                continue
            try:
                similar = self.find_similar(fp, threshold=similarity)
            except Exception:
                similar = []
            # Remove found images from DB immediately to avoid regrouping in later iterations
            try:
                if similar:
                    self.repo.delete_many(similar)
            except Exception:
                pass
            i += 1

            best_score = -1e9
            best_path: Optional[str] = None
            temp_dir = input_dir / str(i)
            temp_dir.mkdir(exist_ok=True)

            for path in similar:
                path_p = Path(path)
                try:
                    score = self.predict_aesthetic(path_p) if use_aesthetics else 0.0
                except Exception:
                    score = 0.0
                if score > best_score:
                    best_score = score
                    best_path = path
                # copy to group folder for inspection
                try:
                    dest = temp_dir / path_p.name
                    if not dest.exists():
                        dest.write_bytes(Path(path).read_bytes())
                except Exception:
                    pass

            if best_path:
                # copy best to output and delete from input
                try:
                    bp = Path(best_path)
                    (output_dir / bp.name).write_bytes(bp.read_bytes())
                    kept.append(best_path)
                    try:
                        bp.unlink()
                    except Exception:
                        pass
                except Exception:
                    pass
            # remove the rest
            for path in similar:
                if path != best_path:
                    try:
                        Path(path).unlink()
                        removed.append(path)
                    except Exception:
                        pass


        return SelectionResult(kept=kept, removed=removed)
