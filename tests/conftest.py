"""Shared fixtures: isolated settings, a fake Hugging Face cache, a service."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from local_image_mcp.config import Settings
from local_image_mcp.registry import Registry, load_registry
from local_image_mcp.resolver import ModelResolver
from local_image_mcp.service import ImageService

FIXED_NOW = datetime(2026, 11, 15, 12, 0, 0, tzinfo=UTC)

CacheWriter = Callable[..., Path]


@pytest.fixture
def hf_cache(tmp_path: Path) -> Path:
    path = tmp_path / "hf-cache"
    path.mkdir()
    return path


@pytest.fixture
def add_cached_file(hf_cache: Path) -> CacheWriter:
    """Create a file in the Hugging Face cache layout and return its snapshot path."""

    def add(
        repo: str, filename: str, content: bytes = b"weights", revision: str = "a" * 40
    ) -> Path:
        repo_dir = hf_cache / f"models--{repo.replace('/', '--')}"
        (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
        (repo_dir / "refs" / "main").write_text(revision)
        target = repo_dir / "snapshots" / revision / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target

    return add


@pytest.fixture
def settings(tmp_path: Path, hf_cache: Path) -> Settings:
    project = tmp_path / "project"
    project.mkdir()
    return Settings(
        state_dir=tmp_path / "state",
        project_dir=project,
        output_dir=project / "generated-images",
        hf_cache_dir=hf_cache,
        sd_binary=str(tmp_path / "no-such-dir" / "sd-cli"),
    )


@pytest.fixture
def registry() -> Registry:
    return load_registry()


@pytest.fixture
def service(settings: Settings, registry: Registry, hf_cache: Path) -> ImageService:
    return ImageService(
        settings, registry=registry, resolver=ModelResolver(hf_cache), clock=lambda: FIXED_NOW
    )
