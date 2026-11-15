from __future__ import annotations

from pathlib import Path

import pytest

from local_image_mcp.errors import RegistryError
from local_image_mcp.registry import FileRef, load_registry
from local_image_mcp.types import Capability


def test_builtin_registry_loads_with_ids() -> None:
    registry = load_registry()
    assert {"z-image-turbo", "flux2-klein-4b", "fake"} <= set(registry.models)
    for model_id, spec in registry.models.items():
        assert spec.id == model_id


def test_builtin_models_match_the_plan() -> None:
    registry = load_registry()
    zit = registry.models["z-image-turbo"]
    assert zit.capabilities == (Capability.TEXT2IMG,)
    assert zit.defaults.steps == 8
    assert zit.defaults.cfg_scale == 1.0
    assert zit.files["diffusion_model"].filename == "z-image-turbo-Q4_K_M.gguf"
    klein = registry.models["flux2-klein-4b"]
    assert set(klein.capabilities) == {Capability.TEXT2IMG, Capability.REFERENCE, Capability.EDIT}
    assert klein.defaults.steps == 4
    assert klein.files["diffusion_model"].repo == "unsloth/FLUX.2-klein-4B-GGUF"


def test_user_registry_overrides_and_removes(tmp_path: Path) -> None:
    user = tmp_path / "models.yaml"
    user.write_text(
        """
schema_version: 1
models:
  flux2-klein-4b: null
  my-model:
    display_name: Mine
    backend: fake
    capabilities: [text2img]
    license: mit
    files:
      diffusion_model: {path: /models/mine.gguf}
""",
        encoding="utf-8",
    )
    registry = load_registry(user)
    assert "flux2-klein-4b" not in registry.models
    assert registry.models["my-model"].files["diffusion_model"] == FileRef(
        path=Path("/models/mine.gguf")
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("backend: nope\n    capabilities: [text2img]", "unknown backend"),
        ("backend: fake\n    capabilities: []", "at least one capability"),
        ("backend: fake\n    capabilities: [reference]", "max_reference_images"),
        ("backend: fake\n    capabilities: [painting]", "capabilities"),
        (
            "backend: fake\n    capabilities: [text2img]\n    defaults: {width: 1000}",
            "default width",
        ),
        (
            "backend: fake\n    capabilities: [text2img]\n    files: {vae: {repo: a/b}}",
            "repo' and 'filename",
        ),
        ("backend: fake\n    capabilities: [text2img]\n    colour: red", "colour"),
    ],
)
def test_invalid_entries_are_rejected(tmp_path: Path, body: str, message: str) -> None:
    user = tmp_path / "models.yaml"
    user.write_text(
        "schema_version: 1\nmodels:\n  bad:\n    display_name: Bad\n    license: mit\n"
        f"    {body}\n",
        encoding="utf-8",
    )
    with pytest.raises(RegistryError, match=message):
        load_registry(user, include_builtin=False)


def test_wrong_schema_version(tmp_path: Path) -> None:
    user = tmp_path / "models.yaml"
    user.write_text("schema_version: 2\nmodels: {}\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="schema_version"):
        load_registry(user)


def test_invalid_yaml(tmp_path: Path) -> None:
    user = tmp_path / "models.yaml"
    user.write_text("models: [unclosed\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="Invalid YAML"):
        load_registry(user)
