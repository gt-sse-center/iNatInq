"""Configuration settings for the app."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """The settings for the app."""

    name: str = "The Inquire API"

    vectordb_collection_name: str
    vectordb_port: str
    vectordb_grpc_port: str

    embedding_model_id: str

    model_config = SettingsConfigDict(env_file="./.env")


@lru_cache
def get_settings():
    """Return the app settings after reading from the .env file."""
    return Settings()
