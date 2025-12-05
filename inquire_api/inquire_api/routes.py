"""Route definitions."""

from typing import Annotated

from fastapi import APIRouter, Form, Request
from loguru import logger

from inquire_api.models import QueryFormData

router = APIRouter()


@router.get("/")
async def index() -> dict[str, str]:
    """Main index page."""
    return {"app": "The Inquire API"}


@router.get("/count")
async def count(request: Request) -> int:
    """Get the number of points in the collection of the dataset."""
    vector_db = request.app.state.vector_db
    return vector_db.client.count(
        collection_name=vector_db.collection_name,
        exact=True,
    ).count


@router.post("/query")
async def process_query(
    request: Request,
    query: Annotated[QueryFormData, Form()],
):
    """Process the search query."""

    logger.info(f"{query=}")

    query_embedding = request.app.state.text_embedder(query.user_input)
    # remove batch dimension
    query_embedding = query_embedding.squeeze()

    vector_db = request.app.state.vector_db

    search_results = vector_db.search(
        query_vector=query_embedding,
        topk=query.k,
        filters=query.filters,
    )

    results = [
        {
            "id": result.id,
            "img_url": result.metadata["img_url"],
            "score": result.score,
            "file_name": result.metadata["file_name"],
            "species": result.metadata["species"],
            "location": result.metadata["location"],
            "observed_on": result.metadata["observed_on"],
        }
        for result in search_results
    ]
    logger.info(f"Number of results: {len(results)}")

    return results
