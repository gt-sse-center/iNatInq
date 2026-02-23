"""Ray task functions for distributed processing.

This module contains the @ray.remote task functions used by both
local Ray and Databricks environments for batch processing.
"""

from .image_processing import process_image_batch_ray

__all__ = [
    "process_image_batch_ray",
]
