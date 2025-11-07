from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Set

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

    # Note: repository is now user-scoped and created within choose_best.
    # db_repo parameter is ignored to avoid accidental cross-user sharing.

        # Lazy-load aesthetics predictor to avoid dependency unless needed
        self._predictor = None
        self._processor = None
        # Progress tracking per user
        # { user_id: { stage: int, percentage: int, eta_seconds: Optional[int], status: str, ... } }
        self._progress: Dict[str, Dict[str, Any]] = {}

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

    def add_image(self, image_path: Path, repo: EmbeddingsRepository) -> None:
        emb = self.embed_image(image_path)
        repo.upsert(str(image_path), emb.tobytes())
        # Note: commit is now done in batch by caller

    def predict_aesthetic(self, image_path: Path) -> float:
        self._ensure_aesthetics()
        img = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._predictor(**inputs)
        return float(outputs.logits[0].item())

    def find_similar(self, query_image: Path, threshold: float, repo: EmbeddingsRepository) -> List[str]:
        """Find similar images using vectorized numpy operations for better performance."""
        q = self.embed_image(query_image)
        paths, embeddings = repo.get_all_as_matrix()
        
        if embeddings is None or len(paths) == 0:
            return []
        
        # Vectorized cosine similarity computation
        similarities = np.dot(embeddings, q)
        similar_indices = np.where(similarities >= threshold)[0]
        
        return [paths[i] for i in similar_indices]

    def choose_best(self, user_id: str, input_dir: Path, output_dir: Path, similarity: float = 0.87, use_aesthetics: bool = True) -> SelectionResult:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare file list once
        files = [input_dir / f for f in os.listdir(input_dir)]
        files = [fp for fp in files if fp.is_file()]
        total = len(files)

        # Create user-scoped embeddings repository
        repo = EmbeddingsRepository(settings.user_db_path(user_id))

        # Stage 1: embeddings indexing with batch commits
        import time as _time
        stage1_start = _time.time()
        processed1 = 0
        self._progress[user_id] = {
            "stage": 1,
            "percentage": 0,
            "eta_seconds": None,
            "status": "indexing",
            "total_stage1": total,
            "processed_stage1": 0,
            "total_stage2": total,
            "processed_stage2": 0,
        }

        # Batch embeddings to reduce commit overhead
        BATCH_SIZE = 10
        for i, fp in enumerate(files):
            try:
                self.add_image(fp, repo)
            except Exception:
                pass
            processed1 += 1
            
            # Commit in batches
            if (i + 1) % BATCH_SIZE == 0 or (i + 1) == total:
                repo.commit()
            
            # Update progress less frequently to reduce overhead
            if processed1 % 5 == 0 or processed1 == total:
                elapsed = max(_time.time() - stage1_start, 1e-6)
                rate = processed1 / elapsed
                remaining = max(total - processed1, 0)
                eta = int(remaining / rate) if rate > 0 else None
                self._progress[user_id].update(
                    {
                        "stage": 1,
                        "percentage": int((processed1 / max(total, 1)) * 100),
                        "eta_seconds": eta,
                        "processed_stage1": processed1,
                    }
                )

        kept: List[str] = []
        removed: List[str] = []
        processed_files: Set[str] = set()  # Track already processed files

        # Stage 2: selection with optimized algorithm
        stage2_start = _time.time()
        self._progress[user_id].update({"stage": 2, "percentage": 0, "eta_seconds": None, "status": "selecting"})

        group_num = 0
        files_processed_count = 0  # Track count for efficient progress updates
        
        for fp in files:
            fp_str = str(fp)
            
            # Skip if already processed as part of another group
            if fp_str in processed_files:
                continue
                
            try:
                similar = self.find_similar(fp, threshold=similarity, repo=repo)
            except Exception:
                similar = []
            
            # If no similar images found, this is a unique image - keep it
            if not similar:
                # Mark as processed
                processed_files.add(fp_str)
                files_processed_count += 1
                # Copy to output as it's unique
                try:
                    dest_path = output_dir / fp.name
                    shutil.copy2(fp, dest_path)
                    kept.append(fp_str)
                    fp.unlink()  # Remove from input after copying
                except Exception:
                    pass
                continue
            
            # Mark all similar images as processed (including the current file)
            processed_files.update(similar)
            files_processed_count += len(similar)
            
            # Remove found images from DB to avoid regrouping in later iterations
            # Batching commits every few groups to reduce transaction overhead
            # Final commit at end ensures any remaining deletes are persisted
            try:
                repo.delete_many(similar)
                # Commit every 5 groups
                if (group_num + 1) % 5 == 0:
                    repo.commit()
            except Exception:
                pass
            
            group_num += 1

            best_score = -1e9
            best_path: Optional[str] = None
            temp_dir = input_dir / str(group_num)
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
                # copy to group folder for inspection using efficient shutil
                try:
                    dest = temp_dir / path_p.name
                    if not dest.exists():
                        shutil.copy2(path_p, dest)
                except Exception:
                    pass

            if best_path:
                # copy best to output and delete from input using efficient operations
                try:
                    bp = Path(best_path)
                    dest_path = output_dir / bp.name
                    shutil.copy2(bp, dest_path)
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
            
            # Update progress for stage 2 - use counter for efficiency
            # Cap at total to avoid exceeding 100%
            current_progress = min(files_processed_count, total)
            if current_progress % 5 == 0 or current_progress >= total:
                elapsed2 = max(_time.time() - stage2_start, 1e-6)
                rate2 = current_progress / elapsed2
                remaining2 = max(total - current_progress, 0)
                eta2 = int(remaining2 / rate2) if rate2 > 0 else None
                self._progress[user_id].update(
                    {
                        "stage": 2,
                        "percentage": int((current_progress / max(total, 1)) * 100),
                        "eta_seconds": eta2,
                        "processed_stage2": current_progress,
                    }
                )

        # Final commit for any pending database operations
        try:
            repo.commit()
        except Exception:
            pass
        
        # Completed
        self._progress[user_id].update({"stage": 2, "percentage": 100, "eta_seconds": 0, "status": "completed"})
        # Ensure DB is closed before returning so the file can be deleted on Windows
        try:
            repo.close()
        except Exception:
            pass
        return SelectionResult(kept=kept, removed=removed)

    def get_progress(self, user_id: str) -> Dict[str, Any]:
        return dict(
            self._progress.get(
                user_id,
                {"stage": 0, "percentage": 0, "eta_seconds": None, "status": "idle"},
            )
        )
