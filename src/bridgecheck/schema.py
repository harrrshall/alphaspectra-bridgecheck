"""Strict public request schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wavelength_nm: list[float] = Field(min_length=2, max_length=5000)
    reflectance: list[float] = Field(min_length=2, max_length=5000)
    reflectance_unit: str = "fraction"
    neighbors: int = Field(default=5, ge=1, le=10)


class AuditSampleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1, max_length=200)
    group_id: str = Field(min_length=1, max_length=200)
    context_wavelength_nm: list[float] = Field(min_length=2, max_length=5000)
    context_reflectance: list[float] = Field(min_length=2, max_length=5000)
    target_wavelength_nm: list[float] = Field(min_length=1, max_length=5000)
    target_reflectance: list[float] = Field(min_length=1, max_length=5000)


class AuditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples: list[AuditSampleRequest] = Field(min_length=2, max_length=10_000)
    bootstrap_repeats: int = Field(default=10_000, ge=100, le=100_000)
