"""Module to handle embedding of data."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class Embedder:
    """Base class to create embeddings."""

    processor: CLIPProcessor
    model: CLIPModel

    def __init__(self, model_id: str) -> None:
        self.device = self.get_device()
        logger.info(f"Using device: {self.device}")
        Embedder.processor = CLIPProcessor.from_pretrained(model_id, use_fast=False)
        Embedder.model = CLIPModel.from_pretrained(model_id).to(self.device)
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

    @staticmethod
    def normalize_features(
        features: torch.Tensor,
        dtype: npt.DTypeLike = np.float32,
    ) -> np.ndarray:
        """Helper method to perform L2 normalization on features and convert it to np.ndarray[dtype]."""
        normalized_features = torch.nn.functional.normalize(features, p=2, dim=1)
        return normalized_features.cpu().numpy().astype(dtype=dtype)


class ImageEmbedder(Embedder):
    """A class to help with creating image embeddings."""

    def __call__(
        self, images: Sequence[Image.Image] | Image.Image, dtype: npt.DTypeLike = np.float32
    ) -> np.ndarray:
        """Forward pass on the image embedder in inference mode.

        Args:
            images (Sequence[Image.Image] | Image.Image): The PIL Image to embed.
            dtype (DTypeLike, optional): The data type of the return numpy array. Defaults to np.float32.

        Returns:
            np.ndarray: The image embedding.
        """
        with torch.inference_mode():
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            feats = self.model.get_image_features(**inputs)

            return self.normalize_features(feats, dtype)


class TextEmbedder(Embedder):
    """A class to help with creating text embeddings."""

    def __call__(self, query: Sequence[str] | str, dtype: npt.DTypeLike = np.float32) -> np.ndarray:
        """Forward pass on the text embedder in inference mode.

        Args:
            query (Sequence[str] | str): The string or sequence of strings to embed.
            dtype (DTypeLike, optional): The data type of the return numpy array. Defaults to np.float32.

        Returns:
            np.ndarray: The text embedding.
        """
        with torch.inference_mode():
            inputs = self.processor(text=query, return_tensors="pt", padding=True).to(self.device)
            feats = self.model.get_text_features(**inputs)

            return self.normalize_features(feats, dtype)
