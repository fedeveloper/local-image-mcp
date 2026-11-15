"""Hugging Face diffusers backend. Interface only; not implemented yet."""

from __future__ import annotations

from local_image_mcp.backends.unavailable import UnavailableBackend


class DiffusersBackend(UnavailableBackend):
    def __init__(self) -> None:
        super().__init__("diffusers", "not implemented yet (planned after the first release)")
