"""Output path policy: where images may be written, and how files are named."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from local_image_mcp.errors import InvalidRequestError, OutputPathError

IMAGE_SUFFIX = ".png"
REFERENCE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class OutputPolicy:
    """Decide the output file for a job.

    - A relative path resolves against the project directory.
    - The final file must be inside one of the allowed roots.
    - A path that ends with a separator, or an existing directory, is a directory:
      the file name is generated.
    - An existing file is never overwritten: a numeric suffix is added.
    """

    def __init__(
        self, project_dir: Path, output_dir: Path, allowed_roots: tuple[Path, ...]
    ) -> None:
        self.project_dir = project_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.allowed_roots = tuple(r.resolve() for r in allowed_roots)

    def output_file(self, requested: str | None, model_id: str, seed: int, now: datetime) -> Path:
        if requested is None or not requested.strip():
            directory, name = self.output_dir, None
        else:
            raw = requested.strip()
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = self.project_dir / path
            is_dir = raw.endswith(("/", "\\")) or path.is_dir() or path.suffix == ""
            if is_dir:
                directory, name = path, None
            else:
                if path.suffix.lower() != IMAGE_SUFFIX:
                    raise OutputPathError(
                        f"Output file must end with '{IMAGE_SUFFIX}', got '{path.name}'."
                    )
                directory, name = path.parent, path.name
        directory = directory.resolve()
        self._check_allowed(directory)
        if name is None:
            name = default_file_name(model_id, seed, now)
        target = unique_output_path(directory / name)
        self._check_allowed(target)
        return target

    def _check_allowed(self, path: Path) -> None:
        for root in self.allowed_roots:
            if path == root or path.is_relative_to(root):
                return
        roots = ", ".join(str(r) for r in self.allowed_roots)
        raise OutputPathError(
            f"Output path {path} is outside the allowed directories ({roots}). "
            "Set LOCAL_IMAGE_ALLOWED_ROOTS to allow more directories."
        )


def default_file_name(model_id: str, seed: int, now: datetime) -> str:
    safe_model = _UNSAFE.sub("-", model_id).strip("-.") or "model"
    return f"{now.strftime('%Y%m%d-%H%M%S')}_{safe_model}_{seed}{IMAGE_SUFFIX}"


def sidecar_path(image_path: Path) -> Path:
    return image_path.with_suffix(".json")


def check_reference_image(path: Path, base_dir: Path) -> Path:
    resolved = (path if path.is_absolute() else base_dir / path).expanduser().resolve()
    if not resolved.is_file():
        raise InvalidRequestError(f"Reference image not found: {resolved}")
    if resolved.suffix.lower() not in REFERENCE_SUFFIXES:
        allowed = ", ".join(sorted(REFERENCE_SUFFIXES))
        raise InvalidRequestError(f"Reference image {resolved.name}: unsupported type ({allowed}).")
    return resolved


def unique_output_path(path: Path) -> Path:
    """Return ``path`` or ``stem-N.suffix`` so that neither the image nor its sidecar exists."""
    candidate = path
    counter = 1
    while candidate.exists() or sidecar_path(candidate).exists():
        candidate = path.with_name(f"{path.stem}-{counter}{path.suffix}")
        counter += 1
    return candidate
