"""Load the built-in registry and merge an optional user registry over it."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import ValidationError

from local_image_mcp.errors import RegistryError
from local_image_mcp.registry.schema import Registry

BUILTIN_REGISTRY = "models.yaml"


def load_registry(user_registry: Path | None = None, *, include_builtin: bool = True) -> Registry:
    """Return the merged registry.

    A model in the user file replaces the built-in model with the same id.
    A model with ``null`` as value in the user file removes the built-in model.
    """
    models: dict[str, Any] = {}
    if include_builtin:
        text = resources.files(__package__).joinpath(BUILTIN_REGISTRY).read_text("utf-8")
        models.update(_parse(text, source="built-in registry")["models"])
    if user_registry is not None:
        try:
            text = user_registry.read_text("utf-8")
        except OSError as exc:
            raise RegistryError(f"Cannot read registry file {user_registry}: {exc}") from exc
        for model_id, entry in _parse(text, source=str(user_registry))["models"].items():
            if entry is None:
                models.pop(model_id, None)
            else:
                models[model_id] = entry
    try:
        return Registry.model_validate({"schema_version": 1, "models": models})
    except ValidationError as exc:
        raise RegistryError(f"Invalid model registry:\n{exc}") from exc


def _parse(text: str, *, source: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RegistryError(f"Invalid YAML in {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise RegistryError(f"{source}: the top level must be a mapping")
    mapping = cast(dict[str, Any], data)
    if mapping.get("schema_version") != 1:
        raise RegistryError(f"{source}: unsupported schema_version (expected 1)")
    entries = mapping.get("models")
    if not isinstance(entries, dict):
        raise RegistryError(f"{source}: 'models' must be a mapping")
    return mapping
