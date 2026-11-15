"""Backend factory."""

from __future__ import annotations

from local_image_mcp.backends.base import Backend, BackendHealth, ProgressCallback
from local_image_mcp.backends.comfyui import ComfyUIBackend
from local_image_mcp.backends.diffusers import DiffusersBackend
from local_image_mcp.backends.fake import FakeBackend
from local_image_mcp.backends.sdcpp import SdCppBackend
from local_image_mcp.config import Settings


def create_backends(settings: Settings) -> dict[str, Backend]:
    """Create one instance of each known backend."""
    return {
        "fake": FakeBackend(),
        "sdcpp": SdCppBackend(
            binary=settings.sd_binary,
            log_dir=settings.state_dir / "logs",
            mode=settings.sd_mode,
            server_binary=settings.sd_server_binary,
            idle_unload_s=settings.sd_server_idle_s,
            server_model_dir=settings.state_dir / "sd-server-models",
        ),
        "diffusers": DiffusersBackend(),
        "comfyui": ComfyUIBackend(),
    }


__all__ = ["Backend", "BackendHealth", "ProgressCallback", "create_backends"]
