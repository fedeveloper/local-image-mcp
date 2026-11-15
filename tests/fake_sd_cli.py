"""A stand-in for stable-diffusion.cpp ``sd-cli``, used by the sdcpp backend tests.

It prints log lines and progress bars in the format of sd-cli commit 532d2c1 and writes
a small PNG. Environment variables select the behaviour:

- ``FAKE_SD_MODE``: ``ok`` (default), ``fail``, ``nofile``, ``hang``.
- ``FAKE_SD_RECORD``: if set, a JSON file where the argv and the prompt are written.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_image_mcp.backends.fake import solid_png


def log(level: str, message: str) -> None:
    stream = sys.stderr if level == "E" else sys.stdout
    stream.write(f"[{level}] {message} --- fake.cpp:1\n")
    stream.flush()


def bar(step: int, total: int, char: str, unit: str) -> None:
    width = 50
    done = step * width // total
    head = ">" if done < width and char == "=" else ""
    fill = char * done + head
    sys.stdout.write(f"\r  |{fill:<{width}}| {step}/{total} - {unit}\x1b[K")
    if step == total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def arg(argv: list[str], *names: str) -> str | None:
    for i, item in enumerate(argv):
        if item in names and i + 1 < len(argv):
            return argv[i + 1]
    return None


def main() -> int:
    argv = sys.argv[1:]
    if "--version" in argv:
        print("stable-diffusion.cpp version master-532d2c1, commit 532d2c1")
        return 0
    if arg(argv, "-M", "--mode") == "metadata":
        print(json.dumps({"schema": "sdcpp.generation/v1", "seed": "1"}))
        return 0

    mode = os.environ.get("FAKE_SD_MODE", "ok")
    prompt_file = arg(argv, "--prompt-file")
    record = os.environ.get("FAKE_SD_RECORD")
    if record:
        prompt = Path(prompt_file).read_text(encoding="utf-8") if prompt_file else None
        Path(record).write_text(json.dumps({"argv": argv, "prompt": prompt}), encoding="utf-8")

    log("I", f"loading diffusion model from '{arg(argv, '--diffusion-model')}'")
    for i in range(1, 5):
        bar(i, 4, "#", "512.00MB/s")
    if mode == "fail":
        log("E", "get sd version from file failed: ''")
        log("I", "sd_ctx_create failed")
        return 1
    width, height = int(arg(argv, "-W") or 64), int(arg(argv, "-H") or 64)
    steps = int(arg(argv, "--steps") or 4)
    log("I", f"generate_image {width}x{height}")
    log("I", "get_learned_condition completed, taking 0.01s")
    log("I", f"generating image: 1/1 - seed {arg(argv, '--seed')}")
    for step in range(1, steps + 1):
        bar(step, steps, "=", "1.00s/it")
        if mode == "hang":
            time.sleep(60)
        time.sleep(0.01)
    log("I", "sampling completed, taking 0.10s")
    log("I", "decoding 1 latents")
    log("I", "decode_first_stage completed, taking 0.01s")
    out = arg(argv, "-o", "--output") or "output.png"
    if mode != "nofile":
        Path(out).write_bytes(solid_png(16, 16, (1, 2, 3)))
    log("I", f"save result image 0 to '{out}' (success)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
