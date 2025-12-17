"""Module with Postgres ORM for performing operations on the dataset."""

from datetime import date

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Date, Double, Integer, Numeric, SmallInteger, String, create_engine
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    """The base class for all declarative ORM models."""


class Photo(Base):
    """Declarative model of the `photos` table."""

    __tablename__ = "photos"

    photo_uuid: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False, primary_key=True)
    photo_id: Mapped[int] = mapped_column(Integer, nullable=False)
    observation_uuid: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    observer_id: Mapped[int] = mapped_column(Integer)
    extension: Mapped[str] = mapped_column(String(5))
    license: Mapped[str] = mapped_column(String(255))
    width: Mapped[int] = mapped_column(SmallInteger)
    height: Mapped[int] = mapped_column(SmallInteger)
    position: Mapped[int] = mapped_column(SmallInteger)
    embedding = mapped_column(Vector)

    def __repr__(self) -> str:
        return f'Photo(photo_uuid="{self.photo_uuid}", photo_id={self.photo_id}, observation_uuid="{self.observation_uuid}", observer_id={self.observer_id},  extension="{self.extension}", license="{self.license}", width={self.width}, height={self.height}, position={self.position})'

    def __str__(self) -> str:
        """String form of object."""
        return f"Photo: id={self.photo_id} [{self.extension}] [{self.width}x{self.height}]"


class Observation(Base):
    """Declarative model of the `observations` table."""

    __tablename__ = "observations"

    observation_uuid: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False, primary_key=True)
    observer_id: Mapped[int] = mapped_column(Integer)
    latitude: Mapped[float] = mapped_column(Numeric(15, 10))
    longitude: Mapped[float] = mapped_column(Numeric(15, 10))
    positional_accuracy: Mapped[int] = mapped_column(Integer)
    taxon_id: Mapped[int] = mapped_column(Integer)
    quality_grade: Mapped[str] = mapped_column(String(255))
    observed_on: Mapped[date] = mapped_column(Date)
    anomaly_score: Mapped[float] = mapped_column(Double)


class Observer(Base):
    """Declarative model of the `observers` table."""

    __tablename__ = "observers"

    observer_id: Mapped[int] = mapped_column(Integer, nullable=False, primary_key=True)
    login: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(255))


class Taxon(Base):
    """Declarative model of the `taxa` table."""

    __tablename__ = "taxa"

    taxon_id: Mapped[int] = mapped_column(Integer, nullable=False, primary_key=True)
    ancestry: Mapped[str] = mapped_column(String(255))
    rank_level: Mapped[float] = mapped_column(Double)
    rank: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(255))
    active: Mapped[bool] = mapped_column(Boolean)

    def __repr__(self) -> str:
        """String representation of a Taxon."""
        return f"Taxon(taxon_id={self.taxon_id}, name={self.name})"

    def __str__(self) -> str:
        """String representation of a Taxon."""
        return f"{self.taxon_id}: {self.name}"


def get_database_session(connection_str: str = "postgresql+psycopg://localhost:5432/inaturalist-open-data"):
    """Yield a database session."""
    engine = create_engine(connection_str)
    with Session(engine) as session:
        yield session
