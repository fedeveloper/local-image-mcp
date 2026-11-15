"""Settings from environment variables.

Environment variables:

- ``LOCAL_IMAGE_STATE_DIR``: where jobs and the GPU lock live.
  Default: the platform user data directory (``%LOCALAPPDATA%\\local-image-mcp`` on Windows).
- ``LOCAL_IMAGE_REGISTRY``: path of a user registry YAML file that adds or overrides models.
  Default: ``<user config dir>/models.yaml`` if it exists.
- ``LOCAL_IMAGE_PROJECT_DIR``: base directory for relative output paths.
  Default: ``CLAUDE_PROJECT_DIR`` (set by Claude Code), else the current directory.
- ``LOCAL_IMAGE_OUTPUT_DIR``: default output directory. Default: ``<project>/generated-images``.
- ``LOCAL_IMAGE_ALLOWED_ROOTS``: extra directories where output files are allowed,
  separated by ``os.pathsep`` (``;`` on Windows, ``:`` on Linux).
- ``LOCAL_IMAGE_SD_BINARY``: path of the stable-diffusion.cpp ``sd-cli`` executable.

The Hugging Face cache location comes from ``HF_HUB_CACHE``, else ``HF_HOME/hub``, else the
``huggingface_hub`` default. ``from_env`` reads these variables itself on each call, because
``huggingface_hub`` reads them only once, at import time.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import platformdirs

APP_NAME = "local-image-mcp"
DEFAULT_OUTPUT_DIR_NAME = "generated-images"


@dataclass(frozen=True)
class Settings:
    state_dir: Path
    project_dir: Path
    output_dir: Path
    extra_allowed_roots: tuple[Path, ...] = ()
    user_registry: Path | None = None
    hf_cache_dir: Path | None = None
    sd_binary: str | None = None

    @property
    def jobs_dir(self) -> Path:
        return self.state_dir / "jobs"

    @property
    def gpu_lock_path(self) -> Path:
        return self.state_dir / "gpu.lock"

    @property
    def allowed_roots(self) -> tuple[Path, ...]:
        roots = [self.project_dir, self.output_dir, *self.extra_allowed_roots]
        unique: list[Path] = []
        for root in roots:
            resolved = root.resolve()
            if resolved not in unique:
                unique.append(resolved)
        return tuple(unique)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, cwd: Path | None = None) -> Settings:
        env = os.environ if env is None else env
        cwd = Path.cwd() if cwd is None else cwd

        state_dir = _path(env.get("LOCAL_IMAGE_STATE_DIR")) or Path(
            platformdirs.user_data_dir(APP_NAME, appauthor=False)
        )
        project_dir = (
            _path(env.get("LOCAL_IMAGE_PROJECT_DIR")) or _path(env.get("CLAUDE_PROJECT_DIR")) or cwd
        )
        output_dir = _path(env.get("LOCAL_IMAGE_OUTPUT_DIR")) or (
            project_dir / DEFAULT_OUTPUT_DIR_NAME
        )
        if not output_dir.is_absolute():
            output_dir = project_dir / output_dir

        roots_value = env.get("LOCAL_IMAGE_ALLOWED_ROOTS", "")
        extra_roots = tuple(Path(p).expanduser() for p in roots_value.split(os.pathsep) if p)

        user_registry = _path(env.get("LOCAL_IMAGE_REGISTRY"))
        if user_registry is None:
            candidate = (
                Path(platformdirs.user_config_dir(APP_NAME, appauthor=False)) / "models.yaml"
            )
            if candidate.is_file():
                user_registry = candidate

        return cls(
            state_dir=state_dir,
            project_dir=project_dir,
            output_dir=output_dir,
            extra_allowed_roots=extra_roots,
            user_registry=user_registry,
            hf_cache_dir=_hf_cache_dir(env),
            sd_binary=env.get("LOCAL_IMAGE_SD_BINARY") or None,
        )


def _hf_cache_dir(env: Mapping[str, str]) -> Path | None:
    """Return the HF hub cache from the environment, or ``None`` for the library default."""
    hub_cache = _path(env.get("HF_HUB_CACHE"))
    if hub_cache is not None:
        return hub_cache
    hf_home = _path(env.get("HF_HOME"))
    if hf_home is not None:
        return hf_home / "hub"
    return None


def _path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser()
