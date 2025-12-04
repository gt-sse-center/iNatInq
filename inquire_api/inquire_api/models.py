"""Pydantic model definitions."""

import datetime
import json

from pydantic import BaseModel, field_validator


class FilterData(BaseModel):
    """Data from the query filters."""

    species: str
    latitude_min: float | None
    latitude_max: float | None
    longitude_min: float | None
    longitude_max: float | None
    start_date: datetime.date | None = None
    end_date: datetime.date | None = None


class QueryFormData(BaseModel):
    """Model representing a query form from a client."""

    user_input: str
    k: int
    filters: FilterData

    @classmethod
    def parse_location(cls, data: dict[str, str | None]) -> dict[str, str | None]:
        """Parse the location data."""
        default_location = {
            "latitudeMin": None,
            "latitudeMax": None,
            "longitudeMin": None,
            "longitudeMax": None,
        }
        location = {}
        for key, v in data.get("location", default_location).items():
            location[key] = float(v) if v else None

        return location

    @field_validator("filters", mode="before")
    @classmethod
    def coerce_to_model(cls, value: str) -> FilterData:
        """Coerce the input for `filters` from a string in the request to the Filters model."""
        d = json.loads(value)

        location = cls.parse_location(d)

        return FilterData(
            species=d.get("species", ""),
            latitude_min=location["latitudeMin"],
            latitude_max=location["latitudeMax"],
            longitude_min=location["longitudeMin"],
            longitude_max=location["longitudeMax"],
        )
