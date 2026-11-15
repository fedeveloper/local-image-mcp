"""A stand-in for stable-diffusion.cpp ``sd-server``, used by the server-mode tests.

It answers the native async API of sd-server commit 532d2c1 (``/sdcpp/v1/...``) and
returns a small PNG. Environment variables select the behaviour:

- ``FAKE_SDS_MODE``: ``ok`` (default), ``exit`` (exits before it listens), ``loaderror``
  (``models/status`` reports ``error``), ``jobfail`` (every job fails), ``stuck``
  (jobs never end and ignore cancel).
- ``FAKE_SDS_RECORD``: if set, a JSON-lines file where the argv and each request body are
  appended.
- ``FAKE_SDS_STEP_S``: seconds per sampling step (default 0.02).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_image_mcp.backends.fake import solid_png

MODE = os.environ.get("FAKE_SDS_MODE", "ok")
STEP_S = float(os.environ.get("FAKE_SDS_STEP_S", "0.02"))
JOBS: dict[str, dict[str, Any]] = {}
LOCK = threading.Lock()


def record(entry: dict[str, Any]) -> None:
    path = os.environ.get("FAKE_SDS_RECORD")
    if path:
        with LOCK, Path(path).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


def arg(argv: list[str], name: str, default: str) -> str:
    return argv[argv.index(name) + 1] if name in argv else default


def job_view(job: dict[str, Any]) -> dict[str, Any]:
    total = int(job["steps"])
    elapsed = time.monotonic() - job["started"]
    view: dict[str, Any] = {"id": job["id"], "status": "generating"}
    if job["cancelled"] and MODE != "stuck":
        view["status"] = "cancelled"
        return view
    if MODE == "stuck":
        view["progress"] = {"stage": "sampling", "completed": 0, "total": total}
        return view
    done = int(elapsed / STEP_S)
    if done < 1:
        view["progress"] = {"stage": "conditioning"}
    elif done <= total:
        view["progress"] = {"stage": "sampling", "completed": done, "total": total}
    elif done <= total + 2:
        view["progress"] = {"stage": "vae_decode"}
    elif MODE == "jobfail":
        view["status"] = "failed"
        view["error"] = {"code": "generation_failed", "message": "fake failure"}
    else:
        png = solid_png(int(job["width"]), int(job["height"]), (200, 80, 40))
        view["status"] = "completed"
        view["result"] = {
            "output_format": "png",
            "images": [
                {
                    "index": 0,
                    "b64_json": base64.b64encode(png).decode("ascii"),
                    "generation_info": {"seed": job["seed"], "steps": total},
                }
            ],
        }
    return view


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write("[I] " + (format % args) + "\n")
        sys.stdout.flush()

    def reply(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/sdcpp/v1/models/status":
            if MODE == "loaderror":
                self.reply(200, {"state": "error", "error": "bad model file"})
            else:
                self.reply(200, {"state": "ready"})
            return
        if self.path.startswith("/sdcpp/v1/jobs/"):
            with LOCK:
                job = JOBS.get(self.path.rsplit("/", 1)[-1])
            if job is None:
                self.reply(404, {"error": "not found"})
            else:
                self.reply(200, job_view(job))
            return
        self.reply(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body: dict[str, Any] = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/sdcpp/v1/img_gen":
            record({"event": "img_gen", "payload": body})
            job_id = uuid.uuid4().hex
            sample: dict[str, Any] = body.get("sample_params") or {}
            with LOCK:
                JOBS[job_id] = {
                    "id": job_id,
                    "started": time.monotonic(),
                    "steps": sample.get("sample_steps", 20),
                    "width": body.get("width", 512),
                    "height": body.get("height", 512),
                    "seed": body.get("seed", -1),
                    "cancelled": False,
                }
            self.reply(202, {"id": job_id, "status": "queued"})
            return
        if self.path.startswith("/sdcpp/v1/jobs/") and self.path.endswith("/cancel"):
            job_id = self.path.split("/")[-2]
            record({"event": "cancel", "id": job_id})
            with LOCK:
                if job_id in JOBS:
                    JOBS[job_id]["cancelled"] = True
            self.reply(200, {"id": job_id, "status": "cancelled"})
            return
        self.reply(404, {"error": "not found"})


def main() -> int:
    argv = sys.argv[1:]
    if "--version" in argv:
        print("stable-diffusion.cpp version master-532d2c1, commit 532d2c1")
        return 0
    record({"event": "start", "argv": argv, "pid": os.getpid()})
    if MODE == "exit":
        print("[E] failed to load model --- fake.cpp:1", flush=True)
        return 1
    host = arg(argv, "--listen-ip", "127.0.0.1")
    port = int(arg(argv, "--listen-port", "1234"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[I] listening on {host}:{port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
