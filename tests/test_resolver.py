from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_image_mcp.errors import InvalidRequestError, MissingModelFilesError
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


def _index(*shards: str) -> bytes:
    return json.dumps({"weight_map": {f"w{i}": s for i, s in enumerate(shards)}}).encode()


SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")


def _cache_unsloth_diffusers(add: CacheWriter, *, missing_shard: bool = False) -> None:
    """The layout found on the Windows test PC (2026-11-15)."""
    add("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q5_K_M.gguf")
    add("unsloth/Z-Image-Turbo", "text_encoder/model.safetensors.index.json", _index(*SHARDS))
    for shard in SHARDS[: 1 if missing_shard else 2]:
        add("unsloth/Z-Image-Turbo", f"text_encoder/{shard}", b"12345")
    add("unsloth/Z-Image-Turbo", "vae/diffusion_pytorch_model.safetensors")


def test_missing_files_are_listed(hf_cache: Path, registry: Registry) -> None:
    status = ModelResolver(hf_cache).status(registry.models["z-image-turbo"])
    assert not status.available
    assert {m.role for m in status.missing} == {"diffusion_model", "text_encoder", "vae"}
    vae = next(m for m in status.missing if m.role == "vae")
    assert vae.source == "unsloth/Z-Image-Turbo/vae/diffusion_pytorch_model.safetensors"
    assert vae.alternatives == [
        "Comfy-Org/z_image_turbo/split_files/vae/ae.safetensors",
        "black-forest-labs/FLUX.1-schnell/ae.safetensors",
    ]


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


def test_other_quantization_and_sharded_text_encoder(
    hf_cache: Path, registry: Registry, add_cached_file: CacheWriter
) -> None:
    _cache_unsloth_diffusers(add_cached_file)
    resolved = ModelResolver(hf_cache).resolve(registry.models["z-image-turbo"])
    assert resolved.files["diffusion_model"].filename == "z-image-turbo-Q5_K_M.gguf"
    te = resolved.files["text_encoder"]
    assert te.path.name == "model.safetensors.index.json"
    assert te.size == len(_index(*SHARDS)) + 2 * 5  # index + both shards


def test_sharded_file_with_missing_shard_is_missing(
    hf_cache: Path, registry: Registry, add_cached_file: CacheWriter
) -> None:
    _cache_unsloth_diffusers(add_cached_file, missing_shard=True)
    status = ModelResolver(hf_cache).status(registry.models["z-image-turbo"])
    assert [m.role for m in status.missing] == ["text_encoder"]


def test_pull_downloads_all_shards(
    hf_cache: Path,
    registry: Registry,
    add_cached_file: CacheWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_cached_file("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf")
    add_cached_file("unsloth/Z-Image-Turbo", "vae/diffusion_pytorch_model.safetensors")
    downloads: list[str] = []

    def fake_download(repo: str, filename: str, **kwargs: object) -> str:
        downloads.append(filename)
        content = _index(*SHARDS) if filename.endswith("index.json") else b"x"
        return str(add_cached_file(repo, filename, content))

    monkeypatch.setattr("local_image_mcp.resolver.hf_hub_download", fake_download)
    resolved = ModelResolver(hf_cache).pull(registry.models["z-image-turbo"])
    assert downloads == [
        "text_encoder/model.safetensors.index.json",
        *(f"text_encoder/{s}" for s in SHARDS),
    ]
    assert resolved.files["text_encoder"].size > 0


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
    add_cached_file("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors")
    downloads: list[tuple[str, str]] = []

    def fake_download(repo: str, filename: str, **kwargs: object) -> str:
        downloads.append((repo, filename))
        return str(add_cached_file(repo, filename))

    monkeypatch.setattr("local_image_mcp.resolver.hf_hub_download", fake_download)
    resolved = ModelResolver(hf_cache).pull(registry.models["z-image-turbo"])
    assert downloads == []  # every role has a cached candidate (alternatives count)
    assert set(resolved.files) == {"diffusion_model", "text_encoder", "vae"}


def test_matches_quant_uses_whole_tokens() -> None:
    from local_image_mcp.resolver import matches_quant

    ref = FileRef(repo="r", filename="sub/z-image-turbo-Q4_K_M.gguf")
    assert matches_quant(ref, "Q4_K_M")
    assert matches_quant(ref, "q4_k_m")
    assert not matches_quant(ref, "Q4_K")
    assert not matches_quant(ref, "K_M")
    assert matches_quant(FileRef(path=Path("/m/flux-Q8_0.gguf")), "Q8_0")


def test_preferred_quant_comes_first_with_fallback(
    hf_cache: Path, registry: Registry, add_cached_file: CacheWriter
) -> None:
    _cache_z_image(add_cached_file)
    add_cached_file("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q8_0.gguf", revision=REV)
    spec = registry.models["z-image-turbo"]

    default = ModelResolver(hf_cache).resolve(spec)
    assert default.files["diffusion_model"].filename == "z-image-turbo-Q4_K_M.gguf"
    preferred = ModelResolver(hf_cache, prefer_quant="Q8_0").resolve(spec)
    assert preferred.files["diffusion_model"].filename == "z-image-turbo-Q8_0.gguf"
    # No Q8_0 text encoder in the registry: the normal order applies.
    assert preferred.files["text_encoder"].filename == "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    fallback = ModelResolver(hf_cache, prefer_quant="Q6_K").resolve(spec)
    assert fallback.files["diffusion_model"].filename == "z-image-turbo-Q4_K_M.gguf"


def test_pull_with_quant_downloads_that_quantization(
    hf_cache: Path,
    registry: Registry,
    add_cached_file: CacheWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cache_z_image(add_cached_file)
    downloads: list[tuple[str, str]] = []

    def fake_download(repo: str, filename: str, **kwargs: object) -> str:
        downloads.append((repo, filename))
        return str(add_cached_file(repo, filename))

    monkeypatch.setattr("local_image_mcp.resolver.hf_hub_download", fake_download)
    resolver = ModelResolver(hf_cache)
    resolved = resolver.pull(registry.models["z-image-turbo"], quant="Q6_K")
    assert downloads == [("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q6_K.gguf")]
    assert resolved.files["diffusion_model"].filename == "z-image-turbo-Q6_K.gguf"

    resolver.pull(registry.models["z-image-turbo"], quant="Q6_K")
    assert len(downloads) == 1  # already cached

    with pytest.raises(InvalidRequestError, match="no file with quantization 'Q2_K'"):
        resolver.pull(registry.models["z-image-turbo"], quant="Q2_K")
