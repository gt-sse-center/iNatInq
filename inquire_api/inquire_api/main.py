"""Main server module."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from inquire_api import routes
from inquire_api.config import get_settings
from inquire_api.embedding import TextEmbedder
from inquire_api.vector_db import VectorDatabaseAdaptor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Function which defines operation at the beginning and the end of the FastAPI app lifespan."""

    # Get the app settings.
    # This pattern allows for overriding `get_settings` for testing.
    app_settings = app.dependency_overrides.get(get_settings, get_settings)()

    # Load the model
    embedder = TextEmbedder(model_id=app_settings.embedding_model_id)
    logger.info("Loaded embedding model.")

    vector_db = VectorDatabaseAdaptor(
        collection_name=app_settings.vectordb_collection_name,
        port=app_settings.vectordb_port,
        grpc_port=app_settings.vectordb_grpc_port,
    )
    logger.info("Loaded vector database.")

    # Assign the embedder to the app state
    app.state.text_embedder = embedder
    app.state.vector_db = vector_db

    yield

    del embedder
    del vector_db


app = FastAPI(title="The Inquire API", lifespan=lifespan)
app.include_router(routes.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,  # Allow cookies
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all HTTP request headers
)
