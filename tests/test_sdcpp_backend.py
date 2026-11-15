from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from local_image_mcp.backends import create_backends
from local_image_mcp.backends.sdcpp import SdCppBackend, find_binary
from local_image_mcp.config import Settings
from local_image_mcp.errors import (
    GenerationFailedError,
    InvalidRequestError,
    JobCancelledError,
)
from local_image_mcp.metadata import read_sidecar
from local_image_mcp.registry import Registry
from local_image_mcp.resolver import ModelResolver
from local_image_mcp.service import ImageService
from local_image_mcp.types import GenerationRequest, ProgressEvent, ProgressStage

from .conftest import FIXED_NOW, CacheWriter

FAKE_CLI = Path(__file__).with_name("fake_sd_cli.py")


@pytest.fixture
def backend() -> SdCppBackend:
    return SdCppBackend(command=[sys.executable, str(FAKE_CLI)])


@pytest.fixture
def sd_service(
    settings: Settings,
    registry: Registry,
    hf_cache: Path,
    add_cached_file: CacheWriter,
    backend: SdCppBackend,
) -> ImageService:
    for repo, filename in [
        ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf"),
        ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
        ("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors"),
        ("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_K_M.gguf"),
        ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
        ("Comfy-Org/flux2-klein-4B", "split_files/vae/flux2-vae.safetensors"),
    ]:
        add_cached_file(repo, filename)
    backends = create_backends(settings)
    backends["sdcpp"] = backend
    return ImageService(
        settings,
        registry=registry,
        resolver=ModelResolver(hf_cache),
        backends=backends,
        clock=lambda: FIXED_NOW,
    )


@pytest.fixture
def record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_SD_RECORD", str(path))
    return path


def _run(service: ImageService, request: GenerationRequest, cancel_after: float | None = None):
    prepared = service.prepare(request)
    events: list[ProgressEvent] = []

    async def scenario():
        cancel = asyncio.Event()
        if cancel_after is not None:
            asyncio.get_running_loop().call_later(cancel_after, cancel.set)
        return await service.run(prepared, events.append, cancel)

    return asyncio.run(scenario()), events


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_health_reports_version(backend: SdCppBackend) -> None:
    health = backend.check()
    assert health.available
    assert health.version == "master-532d2c1 (commit 532d2c1)"
    assert health.detail is None


def test_missing_binary_is_reported(tmp_path: Path) -> None:
    health = SdCppBackend(binary=str(tmp_path / "nope" / "sd-cli.exe")).check()
    assert not health.available
    assert health.detail is not None and "LOCAL_IMAGE_SD_BINARY" in health.detail
    assert find_binary(str(FAKE_CLI)) == str(FAKE_CLI)


def test_z_image_turbo_argv_and_result(sd_service: ImageService, record: Path) -> None:
    prompt = 'A "quoted" title,\nline two, 100% sure, café ☕ & <tags>'
    request = GenerationRequest(
        prompt=prompt, model="z-image-turbo", seed=7, output_path="art/100%_%d.png"
    )
    result, events = _run(sd_service, request)

    recorded: dict[str, Any] = json.loads(record.read_text(encoding="utf-8"))
    argv: list[str] = recorded["argv"]
    assert recorded["prompt"] == prompt  # passed through a UTF-8 file, never the shell
    assert _flag(argv, "--diffusion-model").endswith("z-image-turbo-Q4_K_M.gguf")
    assert _flag(argv, "--llm").endswith("Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
    assert _flag(argv, "--vae").endswith("ae.safetensors")
    assert (_flag(argv, "-W"), _flag(argv, "-H")) == ("1024", "1024")
    assert (_flag(argv, "--steps"), _flag(argv, "--cfg-scale")) == ("8", "1")
    assert _flag(argv, "--seed") == "7"
    assert "--offload-to-cpu" in argv
    assert _flag(argv, "--backend") == "te=cpu"
    assert "-r" not in argv and "--negative-prompt-file" not in argv
    assert "%" not in _flag(argv, "-o")  # sd-cli treats %d in -o as a sequence pattern

    assert result.image_path.name == "100%_%d.png"
    assert result.image_path.read_bytes().startswith(b"\x89PNG")
    assert not list(result.image_path.parent.glob(".local-image-*"))

    steps = [e.step for e in events if e.stage is ProgressStage.SAMPLING]
    assert steps == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert events[-1].stage is ProgressStage.SAVING

    meta = read_sidecar(result.metadata_path)
    assert meta.backend.name == "sdcpp"
    assert meta.backend.version == "master-532d2c1 (commit 532d2c1)"
    assert meta.backend.native_record == {"schema": "sdcpp.generation/v1", "seed": "1"}
    assert meta.backend.argv is not None
    assert _flag(meta.backend.argv, "--prompt-file") == "<prompt>"
    assert _flag(meta.backend.argv, "-o") == str(result.image_path)
    assert meta.model.files["diffusion_model"].repo == "unsloth/Z-Image-Turbo-GGUF"
    assert meta.backend.log_file is None  # no log_dir in this fixture


def test_flux_reference_images(sd_service: ImageService, record: Path) -> None:
    project = sd_service.settings.project_dir
    refs = [project / "a.png", project / "b.jpg"]
    for ref in refs:
        ref.write_bytes(b"img")
    request = GenerationRequest(
        prompt="make it night", model="flux2-klein-4b", seed=1, reference_images=tuple(refs)
    )
    _run(sd_service, request)
    argv: list[str] = json.loads(record.read_text(encoding="utf-8"))["argv"]
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "-r"] == [
        str(r.resolve()) for r in refs
    ]
    assert _flag(argv, "--sampling-method") == "euler"
    assert _flag(argv, "--steps") == "4"
    assert _flag(argv, "--llm").endswith("Qwen3-4B-Q4_K_M.gguf")


def test_raw_output_is_logged(
    sd_service: ImageService, backend: SdCppBackend, tmp_path: Path
) -> None:
    backend.log_dir = tmp_path / "logs"
    result, _ = _run(sd_service, GenerationRequest(prompt="x", model="z-image-turbo", seed=3))
    log = (tmp_path / "logs" / f"{result.image_path.stem}.sd-cli.log").read_text("utf-8")
    assert "generating image: 1/1 - seed 3" in log and "| 8/8 -" in log
    assert read_sidecar(result.metadata_path).backend.log_file == tmp_path / "logs" / (
        f"{result.image_path.stem}.sd-cli.log"
    )


def test_failure_reports_sd_cli_error(
    sd_service: ImageService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SD_MODE", "fail")
    with pytest.raises(GenerationFailedError, match=r"exit code 1.*get sd version from file"):
        _run(sd_service, GenerationRequest(prompt="x", model="z-image-turbo"))
    assert not list(sd_service.settings.output_dir.glob("*.png"))


def test_missing_output_file_is_a_failure(
    sd_service: ImageService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SD_MODE", "nofile")
    with pytest.raises(GenerationFailedError, match="exit code 0"):
        _run(sd_service, GenerationRequest(prompt="x", model="z-image-turbo"))


def test_cancel_stops_the_process(
    sd_service: ImageService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SD_MODE", "hang")
    started = time.monotonic()
    with pytest.raises(JobCancelledError):
        _run(sd_service, GenerationRequest(prompt="x", model="z-image-turbo"), cancel_after=1.0)
    assert time.monotonic() - started < 15


def test_timeout(sd_service: ImageService, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SD_MODE", "hang")
    spec = sd_service.registry.models["z-image-turbo"]
    sd_service.registry.models["z-image-turbo"] = spec.model_copy(
        update={"backend_options": {**spec.backend_options, "timeout_s": 1}}
    )
    with pytest.raises(GenerationFailedError, match="did not finish within 1 s"):
        _run(sd_service, GenerationRequest(prompt="x", model="z-image-turbo"))


def test_unknown_backend_option(sd_service: ImageService) -> None:
    spec = sd_service.registry.models["z-image-turbo"]
    sd_service.registry.models["z-image-turbo"] = spec.model_copy(
        update={"backend_options": {"turbo_mode": True}}
    )
    with pytest.raises(InvalidRequestError, match="turbo_mode"):
        _run(sd_service, GenerationRequest(prompt="x", model="z-image-turbo"))
