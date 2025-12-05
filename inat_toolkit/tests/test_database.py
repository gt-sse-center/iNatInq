"""Tests for the database module."""

from uuid import UUID

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from inat_toolkit.database import Observation, Photo, Taxon


@pytest.fixture(name="connection_str")
def connection_string_fixture():
    return "postgresql+psycopg://localhost:5432/inaturalist-open-data"


@pytest.fixture(name="sqlalchemy_engine")
def sqlalchemy_engine_fixture(connection_str):
    return create_engine(connection_str)


@pytest.fixture(name="session")
def session_fixture(sqlalchemy_engine):
    with Session(sqlalchemy_engine) as session:
        yield session


def test_count(session):
    stmt = select(func.count()).select_from(Photo)
    assert str(stmt) == "SELECT count(*) AS count_1 \nFROM photos"

    result = session.scalar(stmt)
    assert result == 388506299


def test_photo_query(session):
    select_1_photo = select(Photo).limit(1)
    result = session.scalars(select_1_photo).all()

    assert len(result) == 1

    expected_photo = Photo(
        photo_uuid=UUID("6cd74d92-0709-4c7b-b485-7822d95c5d0f"),
        photo_id=119569948,
        observation_uuid=UUID("5c94e2f4-09fb-4f58-9628-1207b751381c"),
        observer_id=2527,
        extension="jpeg",
        license="CC-BY",
        width=2048,
        height=1536,
        position=2,
    )

    photo = result[0]
    print(photo.embedding)
    assert photo.photo_id == expected_photo.photo_id
    assert photo.photo_uuid == expected_photo.photo_uuid
    assert photo.observation_uuid == expected_photo.observation_uuid
    assert photo.observer_id == expected_photo.observer_id
    assert photo.extension == expected_photo.extension
    assert photo.license == expected_photo.license
    assert photo.width == expected_photo.width
    assert photo.height == expected_photo.height
    assert photo.position == expected_photo.position


def test_subset_generation(session):
    stmt = (
        select(
            Photo.photo_id,
            Photo.extension,
            Observation.observer_id,
            Observation.latitude,
            Observation.longitude,
            Observation.positional_accuracy,
            Observation.observed_on,
            Taxon.taxon_id,
            Taxon.name,
        )
        .join(Observation, Photo.observation_uuid == Observation.observation_uuid)
        .join(Taxon, Observation.taxon_id == Taxon.taxon_id)
        .limit(3)
    )

    results = session.execute(stmt).all()

    assert len(results) == 3
