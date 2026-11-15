from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_image_mcp import cli


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    for name in (
        "CLAUDE_PROJECT_DIR",
        "LOCAL_IMAGE_OUTPUT_DIR",
        "LOCAL_IMAGE_REGISTRY",
        "HF_HOME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LOCAL_IMAGE_PROJECT_DIR", str(project))
    monkeypatch.setenv("LOCAL_IMAGE_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    monkeypatch.setenv("LOCAL_IMAGE_SD_BINARY", str(tmp_path / "missing" / "sd-cli"))
    monkeypatch.setattr("local_image_mcp.resolver.hf_hub_download", _no_download)
    return project


def _no_download(*args: object, **kwargs: object) -> str:
    raise AssertionError("tests must not download")


def test_generate_and_regenerate(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["generate", "-p", "a cat", "-m", "fake", "-s", "3", "--json"])
    assert code == cli.EXIT_OK
    job = json.loads(capsys.readouterr().out)
    image = Path(job["result"]["image_path"])
    assert image.is_file() and image.parent == (env / "generated-images").resolve()

    code = cli.main(["regenerate", job["result"]["metadata_path"], "-o", "again.png"])
    assert code == cli.EXIT_OK
    out = capsys.readouterr().out.strip()
    assert Path(out) == (env / "again.png").resolve()
    assert Path(out).read_bytes() == image.read_bytes()


def test_generate_reports_capability_error(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (env / "ref.png").write_bytes(b"png")
    code = cli.main(["generate", "-p", "x", "-m", "z-image-turbo", "-r", "ref.png"])
    assert code == cli.EXIT_ERROR
    assert "does not support reference images" in capsys.readouterr().err


def test_models_list_json(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["models", "list", "--json"]) == cli.EXIT_OK
    models = {m["id"]: m for m in json.loads(capsys.readouterr().out)}
    assert models["flux2-klein-4b"]["capabilities"] == ["text2img", "reference", "edit"]
    assert models["fake"]["files_available"] is True


def test_models_check_exit_code(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["models", "check", "fake"]) == cli.EXIT_OK
    assert cli.main(["models", "check", "z-image-turbo"]) == cli.EXIT_ERROR
    assert "missing  diffusion_model" in capsys.readouterr().out


def test_doctor(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["doctor"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "fake: available" in out and "sdcpp: not available" in out


def test_init_registry_creates_a_valid_template(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_image_mcp.registry.loader import load_registry

    target = tmp_path / "config" / "models.yaml"
    monkeypatch.setenv("LOCAL_IMAGE_REGISTRY", str(target))
    assert cli.main(["models", "init-registry"]) == cli.EXIT_OK
    assert str(target) in capsys.readouterr().out
    registry = load_registry(target)
    assert "z-image-turbo" in registry.models  # the empty template keeps the built-ins

    assert cli.main(["models", "init-registry"]) == cli.EXIT_ERROR
    assert "--force" in capsys.readouterr().err
    assert cli.main(["models", "init-registry", "--force"]) == cli.EXIT_OK

    # The commented example 1 must be a valid registry once it is uncommented.
    lines = target.read_text(encoding="utf-8").splitlines()
    start = lines.index("# models:")
    end = next(i for i, line in enumerate(lines) if line.startswith("# Example 2"))
    example = [line[2:] for line in lines[start:end] if line.startswith("# ")]
    target.write_text("schema_version: 1\n" + "\n".join(example) + "\n", encoding="utf-8")
    spec = load_registry(target).models["my-z-image"]
    assert spec.files["diffusion_model"].path is not None
    assert spec.defaults.steps == 8


def test_pull_unknown_quant_is_an_error(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["models", "pull", "z-image-turbo", "--quant", "Q2_K"]) == cli.EXIT_ERROR
    err = capsys.readouterr().err
    assert "Q2_K" in err and "z-image-turbo-Q8_0.gguf" in err
