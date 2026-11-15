from __future__ import annotations

from pathlib import Path

import pytest

from local_image_mcp.errors import InvalidRequestError, OutputPathError
from local_image_mcp.paths import (
    OutputPolicy,
    check_reference_image,
    default_file_name,
    sidecar_path,
)

from .conftest import FIXED_NOW


@pytest.fixture
def policy(tmp_path: Path) -> OutputPolicy:
    project = tmp_path / "project"
    project.mkdir()
    out = project / "generated-images"
    return OutputPolicy(project, out, (project, out))


def test_default_name_and_directory(policy: OutputPolicy) -> None:
    target = policy.output_file(None, "z-image-turbo", 42, FIXED_NOW)
    assert target.parent == policy.output_dir
    assert target.name == "20261115-120000_z-image-turbo_42.png"


def test_relative_file_resolves_against_project(policy: OutputPolicy) -> None:
    target = policy.output_file("assets/hero.png", "m", 1, FIXED_NOW)
    assert target == policy.project_dir / "assets" / "hero.png"


def test_directory_request_gets_generated_name(policy: OutputPolicy) -> None:
    target = policy.output_file("assets/", "m", 7, FIXED_NOW)
    assert target.parent == policy.project_dir / "assets"
    assert target.name.endswith("_m_7.png")


def test_paths_outside_allowed_roots_are_rejected(policy: OutputPolicy, tmp_path: Path) -> None:
    with pytest.raises(OutputPathError, match="outside the allowed"):
        policy.output_file("../escape.png", "m", 1, FIXED_NOW)
    with pytest.raises(OutputPathError):
        policy.output_file(str(tmp_path / "elsewhere" / "x.png"), "m", 1, FIXED_NOW)


def test_only_png_is_accepted(policy: OutputPolicy) -> None:
    with pytest.raises(OutputPathError, match=r"\.png"):
        policy.output_file("image.jpg", "m", 1, FIXED_NOW)


def test_existing_files_are_not_overwritten(policy: OutputPolicy) -> None:
    first = policy.output_file("a.png", "m", 1, FIXED_NOW)
    first.write_bytes(b"x")
    second = policy.output_file("a.png", "m", 1, FIXED_NOW)
    assert second.name == "a-1.png"
    sidecar_path(second).write_text("{}")  # a sidecar alone also blocks the name
    assert policy.output_file("a.png", "m", 1, FIXED_NOW).name == "a-2.png"


def test_unsafe_model_ids_are_sanitised() -> None:
    assert default_file_name("../evil model", 3, FIXED_NOW) == "20261115-120000_evil-model_3.png"


def test_reference_image_checks(tmp_path: Path) -> None:
    image = tmp_path / "ref.png"
    image.write_bytes(b"png")
    assert check_reference_image(Path("ref.png"), tmp_path) == image.resolve()
    with pytest.raises(InvalidRequestError, match="not found"):
        check_reference_image(Path("missing.png"), tmp_path)
    text = tmp_path / "notes.txt"
    text.write_text("x")
    with pytest.raises(InvalidRequestError, match="unsupported type"):
        check_reference_image(text, tmp_path)
