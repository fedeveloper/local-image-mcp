"""Sidecar JSON metadata, written next to each image.

The sidecar holds everything needed to generate the same image again:
the resolved settings, the exact model files (repo, filename, revision) and the backend.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_image_mcp import __version__
from local_image_mcp.errors import MetadataError
from local_image_mcp.registry.schema import FileRef
from local_image_mcp.types import GenerationRequest, ResolvedFile, ResolvedSettings

SCHEMA_ID = "local-image-mcp/metadata/v1"


class ReferenceImageRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    sha256: str


class FileRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    repo: str | None = None
    filename: str | None = None
    revision: str | None = None
    path: Path | None = None
    size: int

    @classmethod
    def from_resolved(cls, resolved: ResolvedFile) -> FileRecord:
        if resolved.repo is not None:
            return cls(
                repo=resolved.repo,
                filename=resolved.filename,
                revision=resolved.revision,
                size=resolved.size,
            )
        return cls(path=resolved.path, size=resolved.size)

    def to_file_ref(self) -> FileRef:
        if self.repo is not None and self.filename is not None:
            return FileRef(repo=self.repo, filename=self.filename, revision=self.revision)
        if self.path is None:
            raise MetadataError("A file record has neither repo/filename nor path.")
        return FileRef(path=self.path)


class ModelRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    registry_id: str
    display_name: str
    license: str
    files: dict[str, FileRecord]


class BackendRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    argv: list[str] | None = None
    options: dict[str, Any] = Field(default_factory=dict[str, Any])
    native_record: dict[str, Any] | None = None


class RequestRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt: str
    negative_prompt: str | None
    model: str
    width: int
    height: int
    seed: int
    steps: int
    cfg_scale: float
    sampling_method: str | None
    scheduler: str | None
    reference_images: list[ReferenceImageRecord]


class ImageMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_: Literal["local-image-mcp/metadata/v1"] = Field(default=SCHEMA_ID, alias="schema")
    tool_version: str = __version__
    created_at: datetime
    duration_s: float
    image_file: str
    request: RequestRecord
    model: ModelRecord
    backend: BackendRecord
    warnings: list[str] = Field(default_factory=list[str])

    def to_request(self, output_path: str | None = None) -> GenerationRequest:
        """Build a request that generates the same image again."""
        r = self.request
        return GenerationRequest(
            prompt=r.prompt,
            model=r.model,
            width=r.width,
            height=r.height,
            seed=r.seed,
            steps=r.steps,
            cfg_scale=r.cfg_scale,
            negative_prompt=r.negative_prompt,
            reference_images=tuple(ref.path for ref in r.reference_images),
            output_path=output_path,
        )

    def file_overrides(self) -> dict[str, FileRef]:
        """Model files pinned to the recorded repo, filename and revision."""
        return {role: record.to_file_ref() for role, record in self.model.files.items()}


def request_record(settings: ResolvedSettings) -> RequestRecord:
    return RequestRecord(
        prompt=settings.prompt,
        negative_prompt=settings.negative_prompt,
        model=settings.model,
        width=settings.width,
        height=settings.height,
        seed=settings.seed,
        steps=settings.steps,
        cfg_scale=settings.cfg_scale,
        sampling_method=settings.sampling_method,
        scheduler=settings.scheduler,
        reference_images=[
            ReferenceImageRecord(path=p, sha256=sha256_file(p)) for p in settings.reference_images
        ],
    )


def write_sidecar(path: Path, metadata: ImageMetadata) -> None:
    data = metadata.model_dump(mode="json", by_alias=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_sidecar(path: Path) -> ImageMetadata:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MetadataError(f"Cannot read metadata file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MetadataError(f"Metadata file {path} is not valid JSON: {exc}") from exc
    try:
        return ImageMetadata.model_validate(data)
    except ValidationError as exc:
        raise MetadataError(f"Metadata file {path} does not match {SCHEMA_ID}:\n{exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
