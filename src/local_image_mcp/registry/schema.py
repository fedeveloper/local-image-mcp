"""Pydantic schema of the model registry YAML file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from local_image_mcp.types import Capability

KNOWN_BACKENDS = frozenset({"fake", "sdcpp", "diffusers", "comfyui"})


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileRef(_Strict):
    """One model file: a file in a Hugging Face repository, or a local path."""

    repo: str | None = None
    filename: str | None = None
    revision: str | None = None
    path: Path | None = None
    license: str | None = None
    alternatives: tuple[FileRef, ...] = ()

    @model_validator(mode="after")
    def _check_source(self) -> Self:
        has_hub = self.repo is not None or self.filename is not None
        if self.path is not None and has_hub:
            raise ValueError("use either 'path' or 'repo' + 'filename', not both")
        if self.path is None and (self.repo is None or self.filename is None):
            raise ValueError("a file needs 'path', or both 'repo' and 'filename'")
        return self

    def describe(self) -> str:
        if self.path is not None:
            return str(self.path)
        return f"{self.repo}/{self.filename}"


class ModelDefaults(_Strict):
    width: int = 1024
    height: int = 1024
    steps: int = 20
    cfg_scale: float = 1.0
    sampling_method: str | None = None
    scheduler: str | None = None


class ModelLimits(_Strict):
    min_size: int = 256
    max_size: int = 2048
    size_multiple: int = 16
    max_pixels: int | None = None
    min_steps: int = 1
    max_steps: int = 100


class ModelSpec(_Strict):
    id: str = ""
    display_name: str
    backend: str
    capabilities: tuple[Capability, ...]
    license: str
    license_url: str | None = None
    vram_estimate_gb: float | None = None
    supports_negative_prompt: bool = False
    max_reference_images: int = 0
    prompt_notes: str | None = None
    files: dict[str, FileRef] = Field(default_factory=dict[str, FileRef])
    defaults: ModelDefaults = ModelDefaults()
    limits: ModelLimits = ModelLimits()
    backend_options: dict[str, Any] = Field(default_factory=dict[str, Any])

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.backend not in KNOWN_BACKENDS:
            known = ", ".join(sorted(KNOWN_BACKENDS))
            raise ValueError(f"unknown backend '{self.backend}' (known: {known})")
        if not self.capabilities:
            raise ValueError("a model needs at least one capability")
        uses_references = (
            Capability.REFERENCE in self.capabilities or Capability.EDIT in self.capabilities
        )
        if uses_references and self.max_reference_images < 1:
            raise ValueError("'reference' and 'edit' need max_reference_images >= 1")
        if not uses_references and self.max_reference_images != 0:
            raise ValueError("max_reference_images needs the 'reference' or 'edit' capability")
        d, lim = self.defaults, self.limits
        for name, value in (("width", d.width), ("height", d.height)):
            if not lim.min_size <= value <= lim.max_size or value % lim.size_multiple:
                raise ValueError(f"default {name} {value} is outside the model limits")
        if not lim.min_steps <= d.steps <= lim.max_steps:
            raise ValueError(f"default steps {d.steps} is outside the model limits")
        return self

    def has(self, capability: Capability) -> bool:
        return capability in self.capabilities


class Registry(_Strict):
    schema_version: Literal[1]
    models: dict[str, ModelSpec]

    @model_validator(mode="after")
    def _set_ids(self) -> Self:
        for model_id, spec in list(self.models.items()):
            if spec.id and spec.id != model_id:
                raise ValueError(f"model '{model_id}' has a different id '{spec.id}'")
            self.models[model_id] = spec.model_copy(update={"id": model_id})
        return self

    def get(self, model_id: str) -> ModelSpec | None:
        return self.models.get(model_id)

    def with_capability(self, capability: Capability) -> list[str]:
        return [m.id for m in self.models.values() if m.has(capability)]
