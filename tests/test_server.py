"""MCP server tests with an in-process client and the fake backend."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from mcp import Client
from mcp_types import CallToolResult, ImageContent, TextContent

from local_image_mcp.jobs.gpu_lock import GpuLock
from local_image_mcp.jobs.queue import JobQueue
from local_image_mcp.jobs.store import JobStore
from local_image_mcp.server import LocalImageApp, create_server
from local_image_mcp.service import ImageService

Scenario = Callable[[Client], Awaitable[Any]]


def _app(service: ImageService, *, include_fake: bool = True) -> LocalImageApp:
    s = service.settings
    queue = JobQueue(service, JobStore(s.jobs_dir), GpuLock(s.gpu_lock_path, poll_interval=0.01))
    return LocalImageApp(service, queue, include_fake=include_fake)


def _slow_fake(service: ImageService, delay: float) -> None:
    spec = service.registry.models["fake"]
    service.registry.models["fake"] = spec.model_copy(
        update={"backend_options": {"step_delay": delay}}
    )


def _run(app: LocalImageApp, scenario: Scenario) -> Any:
    async def main() -> Any:
        async with Client(create_server(app)) as client:
            return await scenario(client)

    return asyncio.run(main())


def _data(result: CallToolResult) -> dict[str, Any]:
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return result.structured_content


def _error_text(result: CallToolResult) -> str:
    assert result.is_error
    first = result.content[0]
    assert isinstance(first, TextContent)
    return first.text


def test_lists_the_five_tools(service: ImageService) -> None:
    async def scenario(client: Client) -> list[str]:
        return [t.name for t in (await client.list_tools()).tools]

    names = _run(_app(service), scenario)
    assert sorted(names) == [
        "cancel_job",
        "generate_image",
        "get_job_result",
        "get_job_status",
        "list_models",
    ]


def test_list_models_reports_availability(service: ImageService) -> None:
    async def scenario(client: Client) -> dict[str, Any]:
        return _data(await client.call_tool("list_models", {}))

    models = {m["id"]: m for m in _run(_app(service), scenario)["models"]}
    assert models["fake"]["available"] is True
    zimage = models["z-image-turbo"]
    assert zimage["available"] is False
    assert zimage["unavailable_reason"]
    assert zimage["defaults"]["steps"] == 8
    assert "text2img" in zimage["capabilities"]


def test_fake_model_is_hidden_by_default(service: ImageService) -> None:
    async def scenario(client: Client) -> tuple[dict[str, Any], CallToolResult]:
        listed = _data(await client.call_tool("list_models", {}))
        gen = await client.call_tool("generate_image", {"prompt": "x", "model": "fake"})
        return listed, gen

    listed, gen = _run(_app(service, include_fake=False), scenario)
    assert "fake" not in {m["id"] for m in listed["models"]}
    assert "Unknown model 'fake'" in _error_text(gen)


def test_generate_wait_and_result(service: ImageService) -> None:
    async def scenario(client: Client) -> tuple[dict[str, Any], ...]:
        gen = _data(
            await client.call_tool(
                "generate_image", {"prompt": "a lighthouse at dusk", "model": "fake", "seed": 7}
            )
        )
        status = _data(
            await client.call_tool("get_job_status", {"job_id": gen["job_id"], "wait_seconds": 10})
        )
        result = _data(await client.call_tool("get_job_result", {"job_id": gen["job_id"]}))
        return gen, status, result

    gen, status, result = _run(_app(service), scenario)
    assert gen["settings"]["seed"] == 7
    assert gen["settings"]["width"] == 256
    assert gen["settings"]["output_path"].endswith("_fake_7.png")
    assert status["state"] == "succeeded"
    assert status["percent"] == 100.0
    assert "get_job_result" in status["next_step"]
    assert Path(result["image_path"]).is_file()
    assert Path(result["metadata_path"]).is_file()
    assert result["seed"] == 7
    assert result["prompt"] == "a lighthouse at dusk"
    sidecar = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert sidecar["request"]["seed"] == 7


def test_result_with_preview(service: ImageService) -> None:
    async def scenario(client: Client) -> CallToolResult:
        gen = _data(await client.call_tool("generate_image", {"prompt": "x", "model": "fake"}))
        await client.call_tool("get_job_status", {"job_id": gen["job_id"], "wait_seconds": 10})
        return await client.call_tool(
            "get_job_result", {"job_id": gen["job_id"], "include_preview": True}
        )

    result = _run(_app(service), scenario)
    assert not result.is_error
    images = [c for c in result.content if isinstance(c, ImageContent)]
    assert len(images) == 1
    assert images[0].mime_type == "image/jpeg"
    assert len(images[0].data) < 40_000  # well below the token warning


def test_validation_errors_come_back_at_once(service: ImageService, tmp_path: Path) -> None:
    async def scenario(client: Client) -> list[str]:
        calls: list[dict[str, Any]] = [
            {"prompt": "x", "model": "nope"},
            {"prompt": "x", "model": "fake", "width": 100000},
            {"prompt": "x", "model": "fake", "reference_images": [str(tmp_path / "missing.png")]},
            {"prompt": "x", "model": "fake", "output_path": str(tmp_path.parent / "out.png")},
            {"prompt": "x", "model": "z-image-turbo"},
        ]
        return [_error_text(await client.call_tool("generate_image", c)) for c in calls]

    unknown, size, reference, outside, missing = _run(_app(service), scenario)
    assert "Unknown model 'nope'" in unknown
    assert "width" in size.lower()
    assert "Reference image not found" in reference
    assert "outside" in outside.lower() or "allowed" in outside.lower()
    assert "missing" in missing.lower() or "not found" in missing.lower()


def test_reference_on_model_without_capability(service: ImageService, tmp_path: Path) -> None:
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"\x89PNG\r\n\x1a\n")
    fake = service.registry.models["fake"]
    service.registry.models["text-only"] = fake.model_copy(
        update={"id": "text-only", "capabilities": ("text2img",), "max_reference_images": 0}
    )

    async def scenario(client: Client) -> str:
        args = {"prompt": "x", "model": "text-only", "reference_images": [str(ref)]}
        return _error_text(await client.call_tool("generate_image", args))

    message = _run(_app(service), scenario)
    assert "does not support reference images" in message
    assert "fake" in message


def test_status_without_wait_and_cancel(service: ImageService) -> None:
    _slow_fake(service, 0.5)

    async def scenario(client: Client) -> tuple[dict[str, Any], ...]:
        first = _data(await client.call_tool("generate_image", {"prompt": "a", "model": "fake"}))
        second = _data(await client.call_tool("generate_image", {"prompt": "b", "model": "fake"}))
        status = _data(await client.call_tool("get_job_status", {"job_id": second["job_id"]}))
        cancelled_second = _data(await client.call_tool("cancel_job", {"job_id": second["job_id"]}))
        cancelled_first = _data(await client.call_tool("cancel_job", {"job_id": first["job_id"]}))
        no_result = await client.call_tool("get_job_result", {"job_id": first["job_id"]})
        return status, cancelled_second, cancelled_first, {"error": _error_text(no_result)}

    status, second, first, no_result = _run(_app(service), scenario)
    assert status["state"] == "queued"
    assert status["queue_position"] == 1
    assert second["state"] == "cancelled"
    assert first["state"] == "cancelled"
    assert "no image" in no_result["error"]


def test_wait_seconds_returns_a_running_job(service: ImageService) -> None:
    _slow_fake(service, 1.0)

    async def scenario(client: Client) -> dict[str, Any]:
        gen = _data(await client.call_tool("generate_image", {"prompt": "a", "model": "fake"}))
        status = _data(
            await client.call_tool("get_job_status", {"job_id": gen["job_id"], "wait_seconds": 0.3})
        )
        await client.call_tool("cancel_job", {"job_id": gen["job_id"]})
        return status

    status = _run(_app(service), scenario)
    assert status["state"] == "running"
    assert "get_job_status" in status["next_step"]


def test_wait_seconds_above_limit_is_rejected(service: ImageService) -> None:
    async def scenario(client: Client) -> CallToolResult:
        return await client.call_tool("get_job_status", {"job_id": "x", "wait_seconds": 120})

    assert _run(_app(service), scenario).is_error


def test_unknown_job(service: ImageService) -> None:
    async def scenario(client: Client) -> list[str]:
        return [
            _error_text(await client.call_tool(tool, {"job_id": "nope"}))
            for tool in ("get_job_status", "get_job_result", "cancel_job")
        ]

    for message in _run(_app(service), scenario):
        assert "nope" in message


def test_finished_job_survives_a_restart(service: ImageService) -> None:
    async def first(client: Client) -> str:
        gen = _data(await client.call_tool("generate_image", {"prompt": "a", "model": "fake"}))
        await client.call_tool("get_job_status", {"job_id": gen["job_id"], "wait_seconds": 10})
        return str(gen["job_id"])

    job_id = _run(_app(service), first)

    async def second(client: Client) -> dict[str, Any]:
        return _data(await client.call_tool("get_job_result", {"job_id": job_id}))

    result = _run(_app(service), second)
    assert Path(result["image_path"]).is_file()


@pytest.mark.parametrize("value", ["1", "true"])
def test_main_reads_enable_fake(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, bool] = {}

    def fake_from_settings(_settings: object, *, include_fake: bool = False) -> LocalImageApp:
        captured["include_fake"] = include_fake
        raise SystemExit(0)

    monkeypatch.setenv("LOCAL_IMAGE_ENABLE_FAKE", value)
    monkeypatch.setattr(LocalImageApp, "from_settings", staticmethod(fake_from_settings))
    from local_image_mcp import server

    with pytest.raises(SystemExit):
        server.main()
    assert captured["include_fake"] is True
