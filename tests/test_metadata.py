from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_image_mcp.errors import MetadataError
from local_image_mcp.metadata import read_sidecar
from local_image_mcp.service import ImageService
from local_image_mcp.types import GenerationRequest


def test_regenerate_reproduces_the_same_image(service: ImageService) -> None:
    request = GenerationRequest(prompt="a blue whale", model="fake", seed=1234, steps=3)
    first = asyncio.run(service.run(service.prepare(request), lambda _: None, asyncio.Event()))

    meta = read_sidecar(first.metadata_path)
    again = meta.to_request()
    assert again.seed == 1234 and again.steps == 3 and again.prompt == "a blue whale"
    second = asyncio.run(
        service.run(service.prepare(again, meta.file_overrides()), lambda _: None, asyncio.Event())
    )
    assert second.image_path != first.image_path  # never overwrites
    assert second.image_path.read_bytes() == first.image_path.read_bytes()


def test_invalid_sidecar(tmp_path: Path) -> None:
    bad = tmp_path / "x.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(MetadataError, match="not valid JSON"):
        read_sidecar(bad)
    bad.write_text('{"schema": "other/v9"}', encoding="utf-8")
    with pytest.raises(MetadataError, match="does not match"):
        read_sidecar(bad)
    with pytest.raises(MetadataError, match="Cannot read"):
        read_sidecar(tmp_path / "missing.json")
