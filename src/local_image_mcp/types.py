"""Shared value types: capabilities, requests, results."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Capability(StrEnum):
    """What a model can do."""

    TEXT2IMG = "text2img"
    REFERENCE = "reference"
    EDIT = "edit"


class GenerationRequest(BaseModel):
    """A request as the user gives it. `None` means "use the registry default"."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    model: str
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    negative_prompt: str | None = None
    reference_images: tuple[Path, ...] = ()
    output_path: str | None = None


class ResolvedSettings(BaseModel):
    """All generation settings after defaults, validation and seed selection."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    model: str
    width: int
    height: int
    seed: int
    steps: int
    cfg_scale: float
    negative_prompt: str | None = None
    sampling_method: str | None = None
    scheduler: str | None = None
    reference_images: tuple[Path, ...] = ()


class ResolvedFile(BaseModel):
    """A model file found on disk."""

    model_config = ConfigDict(frozen=True)

    role: str
    path: Path
    repo: str | None = None
    filename: str | None = None
    revision: str | None = None
    size: int


class ProgressStage(StrEnum):
    QUEUED = "queued"
    WAITING_FOR_GPU = "waiting_for_gpu"
    LOADING = "loading"
    ENCODING = "encoding"
    SAMPLING = "sampling"
    DECODING = "decoding"
    SAVING = "saving"
    DONE = "done"


class ProgressEvent(BaseModel):
    """A progress update from a backend."""

    model_config = ConfigDict(frozen=True)

    stage: ProgressStage
    step: int | None = None
    total_steps: int | None = None
    message: str | None = None


class BackendOutput(BaseModel):
    """What a backend returns after it wrote the image file."""

    model_config = ConfigDict(frozen=True)

    image_path: Path
    backend_name: str
    backend_version: str
    argv: list[str] | None = None
    options: dict[str, Any] = Field(default_factory=dict[str, Any])
    native_record: dict[str, Any] | None = None
    log_file: Path | None = None


class GenerationResult(BaseModel):
    """The final result of a successful job."""

    model_config = ConfigDict(frozen=True)

    image_path: Path
    metadata_path: Path
    seed: int
    duration_s: float
    backend_name: str
    backend_version: str
    warnings: list[str] = Field(default_factory=list[str])
