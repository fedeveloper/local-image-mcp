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
