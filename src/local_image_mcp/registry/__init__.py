"""Model registry: YAML schema and loader."""

from local_image_mcp.registry.loader import load_registry
from local_image_mcp.registry.schema import (
    KNOWN_BACKENDS,
    FileRef,
    ModelDefaults,
    ModelLimits,
    ModelSpec,
    Registry,
)

__all__ = [
    "KNOWN_BACKENDS",
    "FileRef",
    "ModelDefaults",
    "ModelLimits",
    "ModelSpec",
    "Registry",
    "load_registry",
]
