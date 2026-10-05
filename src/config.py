"""Central configuration, overridable through environment variables (prefix ``PVS_``)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

CANONICAL_SCHEMA: list[str] = [
    "sale_id",
    "sale_date",
    "price",
    "suburb",
    "postcode",
    "state",
    "property_type",
    "bedrooms",
    "bathrooms",
    "parking",
    "land_size_sqm",
    "building_area_sqm",
    "latitude",
    "longitude",
    "year_built",
]


class Settings(BaseSettings):
    """Project settings. Every field can be set via ``PVS_<FIELD_NAME>``."""

    model_config = SettingsConfigDict(env_prefix="PVS_")

    data_dir: str = "data"
    canonical_schema: list[str] = list(CANONICAL_SCHEMA)
    target_col: str = "price"
    test_cutoff_date: str = "2023-01-01"
    mlflow_uri: str = "./mlruns"

    model_name: str = "property-valuation"
    quality_gate_threshold: float = 0.7
    interval_alpha: float = 0.1
    quality_report_path: str = "DATA_QUALITY_REPORT.md"
    reference_data_path: str = "data/reference.parquet"


settings = Settings()
