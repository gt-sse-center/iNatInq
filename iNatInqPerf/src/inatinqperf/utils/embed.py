"""Utilities for embedding images and text using CLIP models."""

from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetInfo, Features, Value, load_from_disk
from datasets import List as HFList
from loguru import logger
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

_EMBED_MATRIX_NDIM = 2


class PretrainedCLIPModel:
    """Helper class for loading and running a pretrained CLIP model."""

    def __init__(self, model_id: str) -> None:
        self.device = self.get_device()
        logger.info(f"Using device: {self.device}")
        self.proc = CLIPProcessor.from_pretrained(model_id, use_fast=False)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        # Get the embedding dimension from the model config
        self.embedding_dim: int = self.model.config.projection_dim

    @staticmethod
    def get_device() -> str:
        """Return the accelerator device which is available."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.mps.is_available():
            return "mps"

        return "cpu"

    def __call__(
        self,
        images: list[Image.Image] | None = None,
        text: list[str] | None = None,
    ) -> np.ndarray:
        """Forward pass of either image or text data."""
        if images is not None:
            inputs = self.proc(images=images, return_tensors="pt", padding=True).to(self.device)
            feats = self.model.get_image_features(**inputs)

        elif text is not None:
            inputs = self.proc(text=text, return_tensors="pt", padding=True).to(self.device)
            feats = self.model.get_text_features(**inputs)

        else:
            msg = "Neither image nor text data provided."
            raise ValueError(msg)

        normalized_feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        return normalized_feats.cpu().numpy().astype(np.float32)


def embed_text(queries: list[str], model_id: str, batch_size: int = 128) -> np.ndarray:
    """Embed text queries using a CLIP model."""
    model = PretrainedCLIPModel(model_id=model_id)

    feats = np.empty((len(queries), model.embedding_dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i : i + batch_size]
            batch_feats = model(text=batch_queries)
            feats[i : i + batch_feats.shape[0]] = batch_feats

    return feats
