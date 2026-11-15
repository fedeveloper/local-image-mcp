from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from local_image_mcp.errors import (
    BackendNotAvailableError,
    CapabilityError,
    InvalidRequestError,
    MissingModelFilesError,
    UnknownModelError,
)
from local_image_mcp.metadata import read_sidecar
from local_image_mcp.service import ImageService
from local_image_mcp.types import GenerationRequest, ProgressEvent, ProgressStage


def _req(**kwargs: object) -> GenerationRequest:
    base: dict[str, object] = {"prompt": "a red fox in the snow", "model": "fake"}
    base.update(kwargs)
    return GenerationRequest.model_validate(base)


def test_defaults_are_applied(service: ImageService) -> None:
    prepared = service.prepare(_req(seed=5))
    s = prepared.settings
    assert (s.width, s.height, s.steps, s.cfg_scale, s.seed) == (256, 256, 4, 1.0, 5)
    assert prepared.output_file.name == "20261115-120000_fake_5.png"


def test_random_seed_is_chosen_and_recorded(service: ImageService) -> None:
    prepared = service.prepare(_req())
    assert 0 <= prepared.settings.seed < 2**32


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"model": "nope"}, UnknownModelError, "Available models"),
        ({"prompt": "   "}, InvalidRequestError, "empty"),
        ({"width": 250}, InvalidRequestError, "multiple of 16"),
        ({"height": 4096}, InvalidRequestError, "outside"),
        ({"steps": 0}, InvalidRequestError, "steps"),
        ({"seed": -1}, InvalidRequestError, "seed"),
        ({"cfg_scale": 99.0}, InvalidRequestError, "cfg_scale"),
    ],
)
def test_invalid_requests(
    service: ImageService, kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        service.prepare(_req(**kwargs))


def test_reference_images_rejected_for_model_without_capability(
    service: ImageService, settings_project_image: Path
) -> None:
    with pytest.raises(CapabilityError) as info:
        service.prepare(_req(model="z-image-turbo", reference_images=[settings_project_image]))
    assert "does not support reference images" in info.value.message
    assert "flux2-klein-4b" in info.value.message


def test_too_many_reference_images(service: ImageService, settings_project_image: Path) -> None:
    with pytest.raises(InvalidRequestError, match="at most 4"):
        service.prepare(_req(reference_images=[settings_project_image] * 5))


def test_negative_prompt_ignored_with_warning(service: ImageService) -> None:
    spec = service.registry.models["fake"]
    service.registry.models["fake"] = spec.model_copy(update={"supports_negative_prompt": False})
    prepared = service.prepare(_req(negative_prompt="blurry"))
    assert prepared.settings.negative_prompt is None
    assert "negative prompt" in prepared.warnings[0]


def test_sdcpp_models_report_backend_not_available(service: ImageService) -> None:
    with pytest.raises(BackendNotAvailableError, match="not found"):
        service.prepare(_req(model="z-image-turbo"))


def test_missing_files_are_reported_when_backend_is_available(service: ImageService) -> None:
    spec = service.registry.models["z-image-turbo"]
    service.registry.models["z-image-turbo"] = spec.model_copy(update={"backend": "fake"})
    with pytest.raises(MissingModelFilesError, match="local-image models pull"):
        service.prepare(_req(model="z-image-turbo"))


def test_run_writes_png_and_sidecar(service: ImageService, settings_project_image: Path) -> None:
    prepared = service.prepare(
        _req(seed=42, reference_images=[settings_project_image], output_path="out/fox.png")
    )
    events: list[ProgressEvent] = []
    result = asyncio.run(service.run(prepared, events.append, asyncio.Event()))

    assert result.image_path == (service.settings.project_dir / "out" / "fox.png").resolve()
    assert result.image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert [e.step for e in events if e.stage is ProgressStage.SAMPLING] == [1, 2, 3, 4]

    raw = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert raw["schema"] == "local-image-mcp/metadata/v1"
    meta = read_sidecar(result.metadata_path)
    assert meta.request.seed == 42
    assert meta.request.reference_images[0].sha256  # inputs are fingerprinted
    assert meta.backend.name == "fake"
    assert meta.image_file == "fox.png"


def test_list_models_reports_readiness(service: ImageService) -> None:
    infos = {i.spec.id: i for i in service.list_models()}
    assert infos["fake"].availability.available and infos["fake"].backend_available
    assert not infos["z-image-turbo"].backend_available
    assert not infos["z-image-turbo"].availability.available


@pytest.fixture
def settings_project_image(service: ImageService) -> Path:
    image = service.settings.project_dir / "ref.png"
    image.write_bytes(b"\x89PNG fake")
    return image


def _png(path: Path, size: tuple[int, int]) -> Path:
    from PIL import Image

    Image.new("RGB", size, (200, 30, 30)).save(path)
    return path


def test_size_follows_reference_when_no_size_given(service: ImageService, tmp_path: Path) -> None:
    ref = _png(tmp_path / "wide.png", (1600, 900))
    s = service.prepare(_req(reference_images=[ref])).settings
    assert s.width > s.height
    assert s.width % 16 == 0 and s.height % 16 == 0
    assert abs(s.width / s.height - 16 / 9) < 0.1
    assert abs(s.width * s.height - 256 * 256) < 256 * 256 * 0.15


def test_explicit_size_wins_over_reference(service: ImageService, tmp_path: Path) -> None:
    ref = _png(tmp_path / "wide.png", (1600, 900))
    s = service.prepare(_req(reference_images=[ref], width=256, height=512)).settings
    assert (s.width, s.height) == (256, 512)


@pytest.mark.parametrize(
    ("ref_size", "expected"),
    [((1024, 1024), (1024, 1024)), ((900, 1600), (768, 1360)), ((3000, 1000), (1776, 592))],
)
def test_size_like_reference_klein_limits(
    tmp_path: Path, ref_size: tuple[int, int], expected: tuple[int, int]
) -> None:
    from local_image_mcp.registry import load_registry
    from local_image_mcp.service import size_like_reference

    spec = load_registry().models["flux2-klein-4b"]
    ref = _png(tmp_path / "ref.png", ref_size)
    w, h = size_like_reference(ref, 1024 * 1024, spec.limits)
    assert (w, h) == expected
    assert spec.limits.max_pixels is not None and w * h <= spec.limits.max_pixels


def test_size_like_reference_unreadable_image_is_square(tmp_path: Path) -> None:
    from local_image_mcp.registry import ModelLimits
    from local_image_mcp.service import size_like_reference

    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    assert size_like_reference(bad, 512 * 512, ModelLimits()) == (512, 512)
