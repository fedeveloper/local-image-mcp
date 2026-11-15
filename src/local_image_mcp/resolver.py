"""Find model files in the local Hugging Face cache.

The resolver never downloads during ``status`` or ``resolve``. Only ``pull`` downloads,
and ``hf_hub_download`` skips files that are already in the cache.
The cache location follows ``HF_HUB_CACHE`` and ``HF_HOME`` (read by ``huggingface_hub``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import cast

from huggingface_hub import (
    hf_hub_download,  # pyright: ignore[reportUnknownVariableType]
    try_to_load_from_cache,
)
from pydantic import BaseModel, ConfigDict

from local_image_mcp.errors import InvalidRequestError, MissingModelFilesError
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
    def __init__(self, cache_dir: Path | None = None, prefer_quant: str | None = None) -> None:
        """``prefer_quant`` (for example ``Q8_0``): candidates with this quantization in the
        file name come first. The other candidates stay as fallbacks."""
        self.cache_dir = cache_dir
        self.prefer_quant = prefer_quant or None

    def candidates(self, ref: FileRef, quant: str | None = None) -> list[FileRef]:
        """The main source, then the alternatives; the preferred quantization first."""
        items = [ref, *ref.alternatives]
        quant = quant or self.prefer_quant
        if not quant:
            return items
        preferred = [c for c in items if matches_quant(c, quant)]
        return preferred + [c for c in items if not matches_quant(c, quant)]

    def find(self, role: str, ref: FileRef) -> ResolvedFile | None:
        """Return the first candidate (see ``candidates``) that is on disk."""
        for candidate in self.candidates(ref):
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
        size = path.stat().st_size
        if is_shard_index(ref.filename):
            shards = shard_files(path)
            if shards is None or not all((path.parent / s).is_file() for s in shards):
                return None
            size += sum((path.parent / s).stat().st_size for s in shards)
        return ResolvedFile(
            role=role,
            path=path,
            repo=ref.repo,
            filename=ref.filename,
            revision=_snapshot_revision(path, ref.filename),
            size=size,
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
        quant: str | None = None,
        on_file: Callable[[str, str], None] | None = None,
    ) -> ResolvedModel:
        """Download the missing files of a model. Files already in the cache are skipped.

        With ``quant``, each file that has a candidate with this quantization uses that
        candidate, even when another quantization is already in the cache.
        """
        if quant and not any(
            matches_quant(c, quant) for ref in spec.files.values() for c in self.candidates(ref)
        ):
            known = sorted(
                {c.describe() for ref in spec.files.values() for c in self.candidates(ref)}
            )
            raise InvalidRequestError(
                f"Model '{spec.id}' has no file with quantization '{quant}'. Known files:\n  "
                + "\n  ".join(known)
                + "\nAdd other files with a user registry (local-image models init-registry)."
            )
        for role, ref in spec.files.items():
            wanted = [c for c in self.candidates(ref, quant) if quant and matches_quant(c, quant)]
            if wanted:
                target = wanted[0]
                if self._find_one(role, target) is not None:
                    continue
            else:
                if self.find(role, ref) is not None:
                    continue
                target = self.candidates(ref, quant)[0]
            self._download(spec, role, target, on_file)
        return ModelResolver(self.cache_dir, quant or self.prefer_quant).resolve(spec)

    def _download(
        self,
        spec: ModelSpec,
        role: str,
        ref: FileRef,
        on_file: Callable[[str, str], None] | None,
    ) -> None:
        if ref.repo is None or ref.filename is None:
            raise MissingModelFilesError(
                f"Model '{spec.id}': local file for '{role}' not found: {ref.describe()}"
            )
        if on_file is not None:
            on_file(role, ref.describe())
        index = hf_hub_download(
            ref.repo, ref.filename, revision=ref.revision, cache_dir=self.cache_dir
        )
        if is_shard_index(ref.filename):
            folder = str(PurePosixPath(ref.filename).parent)
            for shard in shard_files(Path(index)) or []:
                name = shard if folder == "." else f"{folder}/{shard}"
                if on_file is not None:
                    on_file(role, f"{ref.repo}/{name}")
                hf_hub_download(ref.repo, name, revision=ref.revision, cache_dir=self.cache_dir)

    @staticmethod
    def _refs(spec: ModelSpec, overrides: dict[str, FileRef] | None) -> dict[str, FileRef]:
        refs = dict(spec.files)
        if overrides:
            refs.update(overrides)
        return refs


def matches_quant(ref: FileRef, quant: str) -> bool:
    """True if the file name contains ``quant`` as a whole token (``Q8_0``, ``q4_k_m``)."""
    name = PurePosixPath(ref.filename).name if ref.filename else (ref.path.name if ref.path else "")
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(quant)}(?![A-Za-z0-9_])"
    return re.search(pattern, name, flags=re.IGNORECASE) is not None


def is_shard_index(filename: str) -> bool:
    """A ``*.safetensors.index.json`` file lists the shards of one sharded model."""
    return filename.endswith(".safetensors.index.json")


def shard_files(index_path: Path) -> list[str] | None:
    """Return the shard file names listed in a safetensors index, or ``None`` if invalid."""
    try:
        data: object = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    weight_map = cast(dict[str, object], data).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return None
    names = {str(v) for v in cast(dict[str, object], weight_map).values()}
    return sorted(names)


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
