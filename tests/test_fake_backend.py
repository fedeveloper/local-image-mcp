from __future__ import annotations

import struct
import zlib

from local_image_mcp.backends.fake import solid_colour, solid_png


def test_png_is_valid_and_has_the_right_size() -> None:
    data = solid_png(32, 16, (10, 20, 30))
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    length, kind = struct.unpack(">I4s", data[8:16])
    assert kind == b"IHDR" and length == 13
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (32, 16)
    idat_start = data.index(b"IDAT") + 4
    idat_len = struct.unpack(">I", data[idat_start - 8 : idat_start - 4])[0]
    raw = zlib.decompress(data[idat_start : idat_start + idat_len])
    assert raw[:4] == b"\x00\x0a\x14\x1e"
    assert len(raw) == 16 * (1 + 32 * 3)


def test_colour_depends_on_prompt_and_seed() -> None:
    assert solid_colour("a", 1) == solid_colour("a", 1)
    assert solid_colour("a", 1) != solid_colour("a", 2)
