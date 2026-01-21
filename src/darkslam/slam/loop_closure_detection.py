from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

from .feature_encoder import FeatureEncoder


@dataclass(frozen=True)
class LoopClosureConfig:
    detection_threshold: float = 0.98
    id_threshold: int = 250
    num_matches: int = 1


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


class LoopClosureDetection:
    """Loop closure retrieval with cosine similarity (FAISS optional)."""

    def __init__(self, cfg: LoopClosureConfig | None = None) -> None:
        self.cfg = cfg or LoopClosureConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = FeatureEncoder(self.device)

        self._use_faiss = False
        self._faiss_index = None
        try:
            import faiss  # type: ignore

            self._faiss_index = faiss.index_factory(self.model.num_features, "Flat", faiss.METRIC_INNER_PRODUCT)
            self._use_faiss = True
        except ModuleNotFoundError:
            self._faiss_index = None
            self._use_faiss = False

        self._features: List[np.ndarray] = []  # each is (D,)
        self.image_id_to_index: Dict[int, int] = {}
        self.index_to_image_id: Dict[int, int] = {}

    def add(self, image_id: int, image: Tensor) -> None:
        if image.ndim == 3:
            image = image.unsqueeze(dim=0)

        features = self.model(image).detach().cpu().numpy().astype(np.float32)
        features = _l2_normalize(features)
        feat = features[0]

        if image_id in self.image_id_to_index:
            raise ValueError(f"image_id already exists: {image_id}")

        index = len(self._features)
        self._features.append(feat)
        self.image_id_to_index[image_id] = index
        self.index_to_image_id[index] = image_id

        if self._use_faiss:
            assert self._faiss_index is not None
            self._faiss_index.add(features)

    def search(self, image_id: int, *, top_k: int = 100) -> Tuple[List[int], List[float]]:
        if image_id not in self.image_id_to_index:
            raise KeyError(f"Unknown image_id: {image_id}")
        index_id = self.image_id_to_index[image_id]

        if self._use_faiss:
            assert self._faiss_index is not None
            query = np.expand_dims(self._faiss_index.reconstruct(index_id), 0)
            distances, indices = self._faiss_index.search(query, top_k)
            distances = distances.squeeze()
            indices = indices.squeeze()
        else:
            if not self._features:
                return [], []
            query = self._features[index_id][None, :]
            feats = np.stack(self._features, axis=0)
            distances = (feats @ query.T).squeeze(1)  # cosine sim because normalized
            indices = np.arange(len(self._features))
            order = np.argsort(-distances)
            distances = distances[order][:top_k]
            indices = indices[order][:top_k]

        # Filter invalid/self
        mask = indices != -1
        distances = distances[mask]
        indices = indices[mask]

        mask = indices != index_id
        distances = distances[mask]
        indices = indices[mask]

        # Threshold
        mask = distances > self.cfg.detection_threshold
        distances = distances[mask]
        indices = indices[mask]

        # Non-trivial match (skip neighbors)
        mask = np.abs(indices - index_id) > self.cfg.id_threshold
        distances = distances[mask]
        indices = indices[mask]

        distances = distances[: self.cfg.num_matches]
        indices = indices[: self.cfg.num_matches]

        image_ids = sorted([self.index_to_image_id[i] for i in indices.tolist()])
        return image_ids, distances.tolist()

    def predict(self, image_0: Tensor, image_1: Tensor) -> float:
        if image_0.ndim == 3:
            image_0 = image_0.unsqueeze(0)
        if image_1.ndim == 3:
            image_1 = image_1.unsqueeze(0)

        f0 = self.model(image_0).detach().cpu().numpy().astype(np.float32)
        f1 = self.model(image_1).detach().cpu().numpy().astype(np.float32)
        f0 = _l2_normalize(f0)
        f1 = _l2_normalize(f1)
        return float((f0 * f1).sum(axis=1)[0])

