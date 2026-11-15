from __future__ import annotations

import os
from pathlib import Path

from local_image_mcp.config import Settings


def test_claude_project_dir_is_the_default_project(tmp_path: Path) -> None:
    s = Settings.from_env({"CLAUDE_PROJECT_DIR": str(tmp_path)}, cwd=Path("/somewhere"))
    assert s.project_dir == tmp_path
    assert s.output_dir == tmp_path / "generated-images"


def test_explicit_settings(tmp_path: Path) -> None:
    extra = tmp_path / "extra"
    env = {
        "LOCAL_IMAGE_PROJECT_DIR": str(tmp_path),
        "CLAUDE_PROJECT_DIR": "/ignored",
        "LOCAL_IMAGE_OUTPUT_DIR": "art",
        "LOCAL_IMAGE_ALLOWED_ROOTS": os.pathsep.join([str(extra), ""]),
        "LOCAL_IMAGE_STATE_DIR": str(tmp_path / "state"),
        "LOCAL_IMAGE_SD_BINARY": "sd-cli",
    }
    s = Settings.from_env(env, cwd=Path("/nowhere"))
    assert s.output_dir == tmp_path / "art"
    assert s.jobs_dir == tmp_path / "state" / "jobs"
    assert s.sd_binary == "sd-cli"
    assert extra.resolve() in s.allowed_roots
    assert len(s.allowed_roots) == 3


def test_cwd_fallback(tmp_path: Path) -> None:
    s = Settings.from_env({}, cwd=tmp_path)
    assert s.project_dir == tmp_path


def test_hf_cache_from_hub_cache_variable(tmp_path: Path) -> None:
    env = {"HF_HUB_CACHE": str(tmp_path / "hub-cache"), "HF_HOME": str(tmp_path / "home")}
    assert Settings.from_env(env, cwd=tmp_path).hf_cache_dir == tmp_path / "hub-cache"


def test_hf_cache_from_hf_home(tmp_path: Path) -> None:
    env = {"HF_HOME": str(tmp_path / "home")}
    assert Settings.from_env(env, cwd=tmp_path).hf_cache_dir == tmp_path / "home" / "hub"


def test_hf_cache_default_is_library_default(tmp_path: Path) -> None:
    assert Settings.from_env({}, cwd=tmp_path).hf_cache_dir is None


def test_package_version_matches_pyproject() -> None:
    import tomllib

    import local_image_mcp

    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["version"] == local_image_mcp.__version__


def test_preferred_quant(tmp_path: Path) -> None:
    assert Settings.from_env({"LOCAL_IMAGE_QUANT": " Q8_0 "}, cwd=tmp_path).quant == "Q8_0"
    assert Settings.from_env({}, cwd=tmp_path).quant is None
