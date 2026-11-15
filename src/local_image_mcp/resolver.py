"""Find model files in the local Hugging Face cache.

The resolver never downloads during ``status`` or ``resolve``. Only ``pull`` downloads,
and ``hf_hub_download`` skips files that are already in the cache.
The cache location follows ``HF_HUB_CACHE`` and ``HF_HOME`` (read by ``huggingface_hub``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from huggingface_hub import (
    hf_hub_download,  # pyright: ignore[reportUnknownVariableType]
    try_to_load_from_cache,
)
from pydantic import BaseModel, ConfigDict

from local_image_mcp.errors import MissingModelFilesError
from local_image_mcp.registry.schema import FileRef, ModelSpec
from local_image_mcp.types import ResolvedFile


class MissingFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str
    source: str
    alternatives: list[str] = []


class ModelAvailability(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    available: bool
    files: list[ResolvedFile]
    missing: list[MissingFile]


class ResolvedModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    spec: ModelSpec
    files: dict[str, ResolvedFile]


class ModelResolver:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir

    def find(self, role: str, ref: FileRef) -> ResolvedFile | None:
        """Return the first candidate (main source, then alternatives) that is on disk."""
        for candidate in (ref, *ref.alternatives):
            found = self._find_one(role, candidate)
            if found is not None:
                return found
        return None

    def _find_one(self, role: str, ref: FileRef) -> ResolvedFile | None:
        if ref.path is not None:
            path = ref.path.expanduser()
            if path.is_file():
                return ResolvedFile(role=role, path=path.resolve(), size=path.stat().st_size)
            return None
        assert ref.repo is not None and ref.filename is not None
        cached = try_to_load_from_cache(
            ref.repo, ref.filename, cache_dir=self.cache_dir, revision=ref.revision
        )
        if not isinstance(cached, str):
            return None
        path = Path(cached)
        if not path.is_file():
            return None
        return ResolvedFile(
            role=role,
            path=path,
            repo=ref.repo,
            filename=ref.filename,
            revision=_snapshot_revision(path, ref.filename),
            size=path.stat().st_size,
        )

    def status(
        self, spec: ModelSpec, overrides: dict[str, FileRef] | None = None
    ) -> ModelAvailability:
        files: list[ResolvedFile] = []
        missing: list[MissingFile] = []
        for role, ref in self._refs(spec, overrides).items():
            found = self.find(role, ref)
            if found is None:
                missing.append(
                    MissingFile(
                        role=role,
                        source=ref.describe(),
                        alternatives=[a.describe() for a in ref.alternatives],
                    )
                )
            else:
                files.append(found)
        return ModelAvailability(
            model_id=spec.id, available=not missing, files=files, missing=missing
        )

    def resolve(
        self, spec: ModelSpec, overrides: dict[str, FileRef] | None = None
    ) -> ResolvedModel:
        availability = self.status(spec, overrides)
        if availability.missing:
            lines = [f"Model '{spec.id}' is missing files in the local cache:"]
            for item in availability.missing:
                line = f"  - {item.role}: {item.source}"
                if item.alternatives:
                    line += f" (or {', '.join(item.alternatives)})"
                lines.append(line)
            lines.append(f"Download them with: local-image models pull {spec.id}")
            raise MissingModelFilesError("\n".join(lines))
        return ResolvedModel(spec=spec, files={f.role: f for f in availability.files})

    def pull(
        self,
        spec: ModelSpec,
        *,
        on_file: Callable[[str, str], None] | None = None,
    ) -> ResolvedModel:
        """Download the missing files of a model. Files already in the cache are skipped."""
        for role, ref in spec.files.items():
            if self.find(role, ref) is not None:
                continue
            if ref.repo is None or ref.filename is None:
                raise MissingModelFilesError(
                    f"Model '{spec.id}': local file for '{role}' not found: {ref.describe()}"
                )
            if on_file is not None:
                on_file(role, ref.describe())
            hf_hub_download(ref.repo, ref.filename, revision=ref.revision, cache_dir=self.cache_dir)
        return self.resolve(spec)

    @staticmethod
    def _refs(spec: ModelSpec, overrides: dict[str, FileRef] | None) -> dict[str, FileRef]:
        refs = dict(spec.files)
        if overrides:
            refs.update(overrides)
        return refs


def _snapshot_revision(path: Path, filename: str) -> str | None:
    """Return the commit hash from a cache path ``.../snapshots/<commit>/<filename>``."""
    depth = len(Path(filename).parts)
    parents = path.parents
    if len(parents) <= depth:
        return None
    snapshot_dir = parents[depth - 1]
    if snapshot_dir.parent.name != "snapshots":
        return None
    return snapshot_dir.name
