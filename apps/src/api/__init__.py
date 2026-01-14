"""FastAPI routing for the pipeline service.

This package contains all API-specific code:
- HTTP routes and request/response handling
- FastAPI application factory
- Pydantic request/response models
"""

from api import models
from api.app import create_app

__all__ = ["create_app", "models"]
