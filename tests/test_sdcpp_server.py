"""sd-server mode of the sdcpp backend, against ``fake_sd_server.py``."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from local_image_mcp.backends import create_backends, sdcpp
from local_image_mcp.backends.sdcpp import SdCppBackend, find_server_binary
from local_image_mcp.backends.sdserver import progress_event
from local_image_mcp.config import Settings
from local_image_mcp.errors import GenerationFailedError, InvalidRequestError, JobCancelledError
from local_image_mcp.jobs import GpuLock, JobQueue, JobState, JobStore
from local_image_mcp.metadata import read_sidecar
from local_image_mcp.registry import Registry
from local_image_mcp.resolver import ModelResolver
from local_image_mcp.service import ImageService
from local_image_mcp.types import GenerationRequest, ProgressEvent, ProgressStage

from .conftest import FIXED_NOW, CacheWriter

FAKE_CLI = Path(__file__).with_name("fake_sd_cli.py")
FAKE_SERVER = Path(__file__).with_name("fake_sd_server.py")

MakeService = Callable[..., tuple[ImageService, SdCppBackend]]


@pytest.fixture(autouse=True)
def fast_polls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sdcpp, "_JOB_POLL_S", 0.02)


@pytest.fixture
def record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "server-record.jsonl"
    monkeypatch.setenv("FAKE_SDS_RECORD", str(path))
    return path


def _events(record: Path, kind: str) -> list[dict[str, Any]]:
    if not record.exists():
        return []
    lines = record.read_text(encoding="utf-8").splitlines()
    return [e for e in map(json.loads, lines) if e["event"] == kind]


@pytest.fixture
def make_service(
    settings: Settings,
    registry: Registry,
    hf_cache: Path,
    add_cached_file: CacheWriter,
    tmp_path: Path,
) -> MakeService:
    for repo, filename in [
        ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf"),
        ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
        ("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors"),
        ("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_K_M.gguf"),
        ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
        ("Comfy-Org/flux2-klein-4B", "split_files/vae/flux2-vae.safetensors"),
    ]:
        add_cached_file(repo, filename)

    def make(idle_unload_s: float = 120.0) -> tuple[ImageService, SdCppBackend]:
        backend = SdCppBackend(
            command=[sys.executable, str(FAKE_CLI)],
            log_dir=tmp_path / "logs",
            mode="server",
            server_command=[sys.executable, str(FAKE_SERVER)],
            idle_unload_s=idle_unload_s,
            server_model_dir=tmp_path / "sd-server-models",
        )
        backends = create_backends(settings)
        backends["sdcpp"] = backend
        service = ImageService(
            settings,
            registry=registry,
            resolver=ModelResolver(hf_cache),
            backends=backends,
            clock=lambda: FIXED_NOW,
        )
        return service, backend

    return make


async def _generate(
    service: ImageService,
    request: GenerationRequest,
    events: list[ProgressEvent] | None = None,
    cancel: asyncio.Event | None = None,
):
    prepared = service.prepare(request)
    sink: list[ProgressEvent] = events if events is not None else []
    return await service.run(prepared, sink.append, cancel or asyncio.Event())


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_health_names_the_server_mode(make_service: MakeService) -> None:
    _, backend = make_service()
    health = backend.check()
    assert health.available
    assert health.version is not None and health.version.endswith("sd-server mode")


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(InvalidRequestError):
        SdCppBackend(mode="daemon")


def test_find_server_binary(tmp_path: Path) -> None:
    name = "sd-server.exe" if sys.platform == "win32" else "sd-server"
    cli = tmp_path / ("sd-cli.exe" if sys.platform == "win32" else "sd-cli")
    server = tmp_path / name
    cli.write_text("")
    server.write_text("")
    assert find_server_binary(None, str(cli)) == str(server)
    other = tmp_path / "other-server"
    other.write_text("")
    assert find_server_binary(str(other), str(cli)) == str(other)


def test_server_mode_generates_and_keeps_the_model(make_service: MakeService, record: Path) -> None:
    service, backend = make_service()
    events: list[ProgressEvent] = []

    async def scenario() -> tuple[Any, Any, bool]:
        first = await _generate(
            service, GenerationRequest(prompt="a fox", model="z-image-turbo", seed=7), events
        )
        second = await _generate(
            service, GenerationRequest(prompt="a cat", model="z-image-turbo", seed=8)
        )
        running = backend.server_running
        await service.aclose()
        return first, second, running

    first, second, running = asyncio.run(scenario())
    assert running
    assert not backend.server_running

    starts = _events(record, "start")
    assert len(starts) == 1  # the second image reuses the loaded model
    argv: list[str] = starts[0]["argv"]
    assert _flag(argv, "--diffusion-model").endswith("z-image-turbo-Q4_K_M.gguf")
    assert _flag(argv, "--llm").endswith("Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
    assert _flag(argv, "--listen-ip") == "127.0.0.1"
    assert int(_flag(argv, "--listen-port")) > 0
    assert "--model-dir" in argv

    payloads = [e["payload"] for e in _events(record, "img_gen")]
    assert [p["prompt"] for p in payloads] == ["a fox", "a cat"]
    assert payloads[0]["seed"] == 7
    assert (payloads[0]["width"], payloads[0]["height"]) == (1024, 1024)
    assert payloads[0]["sample_params"]["sample_steps"] == 8
    assert payloads[0]["sample_params"]["guidance"]["txt_cfg"] == 1.0
    assert payloads[0]["ref_images"] == []

    with Image.open(first.image_path) as img:
        assert img.size == (1024, 1024)
    assert second.image_path != first.image_path
    meta = read_sidecar(first.metadata_path)
    assert meta.backend.options["mode"] == "server"
    assert meta.backend.native_record == {"seed": 7, "steps": 8}
    sampling = [e for e in events if e.stage is ProgressStage.SAMPLING]
    assert sampling and sampling[-1].total_steps == 8
    assert not list(first.image_path.parent.glob(".local-image-*"))


def test_model_change_restarts_the_server(
    make_service: MakeService, record: Path, tmp_path: Path
) -> None:
    service, _ = make_service()
    ref = tmp_path / "ref.png"
    Image.new("RGB", (64, 64), (10, 120, 200)).save(ref)

    async def scenario() -> None:
        await _generate(service, GenerationRequest(prompt="a", model="z-image-turbo"))
        await _generate(
            service,
            GenerationRequest(
                prompt="make it red", model="flux2-klein-4b", reference_images=(ref,)
            ),
        )
        await service.aclose()

    asyncio.run(scenario())
    starts = _events(record, "start")
    assert len(starts) == 2
    assert _flag(starts[1]["argv"], "--diffusion-model").endswith("flux-2-klein-4b-Q4_K_M.gguf")
    payload = _events(record, "img_gen")[1]["payload"]
    assert base64.b64decode(payload["ref_images"][0]) == ref.read_bytes()


def test_idle_unload_after_the_delay(make_service: MakeService, record: Path) -> None:
    service, backend = make_service(idle_unload_s=0.3)

    async def scenario() -> list[bool]:
        seen: list[bool] = []
        await _generate(service, GenerationRequest(prompt="a", model="z-image-turbo"))
        await service.on_queue_empty()
        await asyncio.sleep(0.1)
        seen.append(backend.server_running)  # still in the grace period
        await _generate(service, GenerationRequest(prompt="b", model="z-image-turbo"))
        await asyncio.sleep(0.4)
        seen.append(backend.server_running)  # the new job cancelled the timer
        await service.on_queue_empty()
        for _ in range(150):
            if not backend.server_running:
                break
            await asyncio.sleep(0.02)
        seen.append(backend.server_running)
        await service.aclose()
        return seen

    assert asyncio.run(scenario()) == [True, True, False]
    assert len(_events(record, "start")) == 1


def test_queue_unloads_when_it_is_empty(make_service: MakeService, record: Path) -> None:
    service, backend = make_service(idle_unload_s=0)
    s = service.settings

    async def scenario() -> list[JobState]:
        queue = JobQueue(service, JobStore(s.jobs_dir), GpuLock(s.gpu_lock_path, 0.01))
        jobs = [
            queue.submit(GenerationRequest(prompt=p, model="z-image-turbo")) for p in ("a", "b")
        ]
        async with queue:
            done = [await queue.wait(j.id, timeout=30) for j in jobs]
            for _ in range(100):
                if not backend.server_running:
                    break
                await asyncio.sleep(0.02)
            assert not backend.server_running
        return [j.state for j in done]

    assert asyncio.run(scenario()) == [JobState.SUCCEEDED, JobState.SUCCEEDED]
    assert len(_events(record, "start")) == 1  # one load for two queued jobs


def test_queue_stop_closes_the_server(make_service: MakeService) -> None:
    service, backend = make_service()
    s = service.settings

    async def scenario() -> bool:
        queue = JobQueue(service, JobStore(s.jobs_dir), GpuLock(s.gpu_lock_path, 0.01))
        async with queue:
            job = queue.submit(GenerationRequest(prompt="a", model="z-image-turbo"))
            await queue.wait(job.id, timeout=30)
            running = backend.server_running
        return running

    assert asyncio.run(scenario())  # kept loaded during the grace period
    assert not backend.server_running


@pytest.mark.parametrize(
    ("mode", "message"),
    [("exit", "stopped while it loaded"), ("loaderror", "bad model file")],
)
def test_start_failures(
    make_service: MakeService, monkeypatch: pytest.MonkeyPatch, mode: str, message: str
) -> None:
    monkeypatch.setenv("FAKE_SDS_MODE", mode)
    service, backend = make_service()

    async def scenario() -> None:
        await _generate(service, GenerationRequest(prompt="a", model="z-image-turbo"))

    with pytest.raises(GenerationFailedError, match=message):
        asyncio.run(scenario())
    assert not backend.server_running


def test_job_failure(make_service: MakeService, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SDS_MODE", "jobfail")
    service, _ = make_service()

    async def scenario() -> None:
        try:
            await _generate(service, GenerationRequest(prompt="a", model="z-image-turbo"))
        finally:
            await service.aclose()

    with pytest.raises(GenerationFailedError, match="fake failure"):
        asyncio.run(scenario())


def test_cancel_sends_the_cancel_request(
    make_service: MakeService, monkeypatch: pytest.MonkeyPatch, record: Path
) -> None:
    monkeypatch.setenv("FAKE_SDS_STEP_S", "0.2")
    service, backend = make_service()

    async def scenario() -> bool:
        cancel = asyncio.Event()
        asyncio.get_running_loop().call_later(0.8, cancel.set)
        try:
            await _generate(
                service, GenerationRequest(prompt="a", model="z-image-turbo"), cancel=cancel
            )
        finally:
            running = backend.server_running
            await service.aclose()
        return running

    with pytest.raises(JobCancelledError):
        asyncio.run(scenario())
    assert len(_events(record, "cancel")) == 1


def test_stuck_job_kills_the_server(
    make_service: MakeService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SDS_MODE", "stuck")
    monkeypatch.setattr(sdcpp, "_CANCEL_WAIT_S", 0.3)
    service, backend = make_service()

    async def scenario() -> None:
        cancel = asyncio.Event()
        asyncio.get_running_loop().call_later(0.8, cancel.set)
        await _generate(
            service, GenerationRequest(prompt="a", model="z-image-turbo"), cancel=cancel
        )

    with pytest.raises(JobCancelledError):
        asyncio.run(scenario())
    assert not backend.server_running


def test_progress_event_mapping() -> None:
    sampling = {"progress": {"stage": "sampling", "completed": 3, "total": 8}}
    assert progress_event(sampling) == ProgressEvent(
        stage=ProgressStage.SAMPLING, step=3, total_steps=8
    )
    decode = progress_event({"progress": {"stage": "vae_decode"}})
    assert decode == ProgressEvent(stage=ProgressStage.DECODING)
    assert progress_event({"progress": {"stage": "something_new"}}) is None
    assert progress_event({}) is None


def test_settings_read_the_server_variables(tmp_path: Path) -> None:
    env = {
        "LOCAL_IMAGE_SD_MODE": "Server",
        "LOCAL_IMAGE_SD_SERVER_BINARY": "/opt/sd/sd-server",
        "LOCAL_IMAGE_SD_SERVER_IDLE": "0",
    }
    s = Settings.from_env(env, cwd=tmp_path)
    assert (s.sd_mode, s.sd_server_binary, s.sd_server_idle_s) == ("server", "/opt/sd/sd-server", 0)
    defaults = Settings.from_env({}, cwd=tmp_path)
    assert (defaults.sd_mode, defaults.sd_server_idle_s) == ("auto", 120.0)
    with pytest.raises(InvalidRequestError):
        Settings.from_env({"LOCAL_IMAGE_SD_MODE": "daemon"}, cwd=tmp_path)
    with pytest.raises(InvalidRequestError):
        Settings.from_env({"LOCAL_IMAGE_SD_SERVER_IDLE": "soon"}, cwd=tmp_path)


def test_auto_mode_picks_server_when_it_is_found(tmp_path: Path) -> None:
    exe = ".exe" if sys.platform == "win32" else ""
    cli = tmp_path / f"sd-cli{exe}"
    cli.write_text("")
    assert SdCppBackend(binary=str(cli)).effective_mode() == "cli"
    (tmp_path / f"sd-server{exe}").write_text("")
    assert SdCppBackend(binary=str(cli)).effective_mode() == "server"
    assert SdCppBackend(binary=str(cli), mode="cli").effective_mode() == "cli"
    fake = SdCppBackend(command=[sys.executable, str(FAKE_CLI)])
    assert fake.effective_mode() == "cli"
