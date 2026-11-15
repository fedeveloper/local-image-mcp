"""stable-diffusion.cpp backend (``sd-cli`` subprocess).

Placeholder in Phase 1. Phase 2 implements it against a verified ``sd-cli`` version.
"""

from __future__ import annotations

from local_image_mcp.backends.unavailable import UnavailableBackend


class SdCppBackend(UnavailableBackend):
    def __init__(self, binary: str | None = None) -> None:
        super().__init__("sdcpp", "not implemented yet (Phase 2)")
        self.binary = binary
