"""Module to handle embedding of query data."""

import numpy as np
import torch
from loguru import logger
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class Embedder:
    """Base class to create embeddings."""

    def __init__(self, model_id: str) -> None:
        self.device = self.get_device()
        logger.info(f"Using device: {self.device}")
        self.proc = CLIPProcessor.from_pretrained(model_id, use_fast=False)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        # Get the embedding dimension from the model config
        self.embedding_dim: int = self.model.config.projection_dim

    @staticmethod
    def get_device() -> torch.device:
        """Get the accelerator device for running the torch model."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


class ImageEmbedder(Embedder):
    """A class to help with creating image embeddings."""

    def __call__(self, image: Image.Image) -> np.ndarray:
        """Forward pass on the image embedder in inference mode."""
        with torch.inference_mode():
            inputs = self.proc(images=image, return_tensors="pt", padding=True).to(self.device)
            feats = self.model.get_image_features(**inputs)

            normalized_feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            return normalized_feats.cpu().numpy().astype(np.float32)


class TextEmbedder(Embedder):
    """A class to help with creating text embeddings."""

    def __call__(self, query: str) -> np.ndarray:
        """Forward pass on the text embedder in inference mode."""
        with torch.inference_mode():
            inputs = self.proc(text=query, return_tensors="pt", padding=True).to(self.device)
            feats = self.model.get_text_features(**inputs)

            normalized_feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            return normalized_feats.cpu().numpy().astype(np.float32)
