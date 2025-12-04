"""Configuration settings for the app."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def get_env_filepath():
    """Get the absolute path to the .env file."""
    return Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    """The settings for the app."""

    name: str = "The Inquire API"

    vectordb_collection_name: str
    vectordb_port: str
    vectordb_grpc_port: str

    embedding_model_id: str

    model_config = SettingsConfigDict(env_file=get_env_filepath())


@lru_cache
def get_settings():
    """Return the app settings after reading from the .env file."""
    return Settings()
