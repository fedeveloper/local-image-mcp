from __future__ import annotations

from pathlib import Path

import pytest

from local_image_mcp.errors import MissingModelFilesError
from local_image_mcp.registry import FileRef, Registry
from local_image_mcp.resolver import ModelResolver

from .conftest import CacheWriter

REV = "b" * 40


def _cache_z_image(add: CacheWriter, *, vae_from_alternative: bool = False) -> None:
    add("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf", revision=REV)
    add("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
    if vae_from_alternative:
        add("black-forest-labs/FLUX.1-schnell", "ae.safetensors")
    else:
        add("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors", revision=REV)


def test_missing_files_are_listed(hf_cache: Path, registry: Registry) -> None:
    status = ModelResolver(hf_cache).status(registry.models["z-image-turbo"])
    assert not status.available
    assert {m.role for m in status.missing} == {"diffusion_model", "text_encoder", "vae"}
    vae = next(m for m in status.missing if m.role == "vae")
    assert vae.alternatives == ["black-forest-labs/FLUX.1-schnell/ae.safetensors"]


def test_resolve_finds_cached_files_with_revision(
    hf_cache: Path, registry: Registry, add_cached_file: CacheWriter
) -> None:
    _cache_z_image(add_cached_file)
    resolved = ModelResolver(hf_cache).resolve(registry.models["z-image-turbo"])
    diffusion = resolved.files["diffusion_model"]
    assert diffusion.path.name == "z-image-turbo-Q4_K_M.gguf"
    assert diffusion.revision == REV
    assert diffusion.size == len(b"weights")
    vae = resolved.files["vae"]  # nested file name in the repository
    assert vae.revision == REV
    assert vae.path.parts[-3:] == ("split_files", "vae", "ae.safetensors")


def test_alternative_source_is_used(
    hf_cache: Path, registry: Registry, add_cached_file: CacheWriter
) -> None:
    _cache_z_image(add_cached_file, vae_from_alternative=True)
    resolved = ModelResolver(hf_cache).resolve(registry.models["z-image-turbo"])
    assert resolved.files["vae"].repo == "black-forest-labs/FLUX.1-schnell"


def test_resolve_error_names_files_and_pull_command(hf_cache: Path, registry: Registry) -> None:
    with pytest.raises(MissingModelFilesError) as info:
        ModelResolver(hf_cache).resolve(registry.models["flux2-klein-4b"])
    message = info.value.message
    assert "unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q4_K_M.gguf" in message
    assert "local-image models pull flux2-klein-4b" in message


def test_local_path_and_overrides(
    tmp_path: Path, hf_cache: Path, registry: Registry, add_cached_file: CacheWriter
) -> None:
    _cache_z_image(add_cached_file)
    local = tmp_path / "custom.gguf"
    local.write_bytes(b"1234")
    resolved = ModelResolver(hf_cache).resolve(
        registry.models["z-image-turbo"], {"diffusion_model": FileRef(path=local)}
    )
    assert resolved.files["diffusion_model"].path == local.resolve()
    assert resolved.files["diffusion_model"].repo is None


def test_pinned_revision_that_is_not_cached_is_missing(
    hf_cache: Path, registry: Registry, add_cached_file: CacheWriter
) -> None:
    _cache_z_image(add_cached_file)
    pinned = FileRef(
        repo="unsloth/Z-Image-Turbo-GGUF", filename="z-image-turbo-Q4_K_M.gguf", revision="c" * 40
    )
    status = ModelResolver(hf_cache).status(
        registry.models["z-image-turbo"], {"diffusion_model": pinned}
    )
    assert [m.role for m in status.missing] == ["diffusion_model"]


def test_pull_skips_cached_files(
    hf_cache: Path,
    registry: Registry,
    add_cached_file: CacheWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_cached_file("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf")
    add_cached_file("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
    downloads: list[tuple[str, str]] = []

    def fake_download(repo: str, filename: str, **kwargs: object) -> str:
        downloads.append((repo, filename))
        return str(add_cached_file(repo, filename))

    monkeypatch.setattr("local_image_mcp.resolver.hf_hub_download", fake_download)
    resolved = ModelResolver(hf_cache).pull(registry.models["z-image-turbo"])
    assert downloads == [("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors")]
    assert set(resolved.files) == {"diffusion_model", "text_encoder", "vae"}
