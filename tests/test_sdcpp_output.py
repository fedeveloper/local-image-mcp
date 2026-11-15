from __future__ import annotations

from local_image_mcp.backends.sdcpp_output import SdCliOutputParser, parse_version
from local_image_mcp.types import ProgressEvent, ProgressStage

# Output in the format of sd-cli commit 532d2c1 (see src/core/util.cpp, examples/common/log.cpp).
SAMPLE = (
    "[I] loading diffusion model from 'z.gguf' --- diffusion_engine.cpp:697\n"
    "\r  |#########################>                        | 400/800 - 512.00MB/s\x1b[K"
    "\r  |##################################################| 800/800 - 515.00MB/s\x1b[K\n"
    "[I] generate_image 1024x1024 --- image.cpp:954\n"
    "[I] get_learned_condition completed, taking 2.10s --- image.cpp:664\n"
    "[I] generating image: 1/1 - seed 42 --- image.cpp:1005\n"
    "\r  |======>                                           | 1/8 - 3.20s/it - ETA 00:22\x1b[K"
    "\r  |============>                                     | 2/8 - 3.10s/it - ETA 00:19\x1b[K"
    "\r  |==================================================| 8/8 - 3.00s/it\x1b[K\n"
    "[I] sampling completed, taking 24.50s --- image.cpp:1038\n"
    "\r  |=========================>                        | 2/4 - 1.00it/s\x1b[K"
    "[I] decoding 1 latents --- image.cpp:689\n"
    "[I] decode_first_stage completed, taking 1.20s --- image.cpp:759\n"
    "[W] something odd --- x.cpp:1\n"
    "[I] save result image 0 to 'out.png' (success) --- main.cpp:501\n"
    "[I] generate_image completed in 30.10s --- image.cpp:1227\n"
)


def _events(chunks: list[str]) -> list[tuple[ProgressStage, int | None]]:
    parser = SdCliOutputParser(total_steps=8)
    events: list[ProgressEvent] = []
    for chunk in chunks:
        events += parser.feed(chunk)
    events += parser.finish()
    return [(e.stage, e.step) for e in events]


def test_stages_and_steps() -> None:
    assert _events([SAMPLE]) == [
        (ProgressStage.ENCODING, None),
        (ProgressStage.SAMPLING, 0),
        (ProgressStage.SAMPLING, 1),
        (ProgressStage.SAMPLING, 2),
        (ProgressStage.SAMPLING, 8),
        (ProgressStage.DECODING, None),
        (ProgressStage.SAVING, None),
    ]


def test_chunk_boundaries_do_not_matter() -> None:
    one_char_chunks = list(SAMPLE)
    assert _events(one_char_chunks) == _events([SAMPLE])


def test_stdout_and_stderr_chunks_do_not_mix() -> None:
    parser = SdCliOutputParser(total_steps=8)
    events = parser.feed(
        "[I] generating image: 1/1 - seed 1 --- a.cpp:1\n\r  |===>   | 7/", "stdout"
    )
    events += parser.feed("ggml_cuda_init: found 1 CUDA device\n", "stderr")
    events += parser.feed("8 - 1.00s/it\x1b[K\r  |=======| 8/8 - 1.00s/it\x1b[K\n", "stdout")
    assert [(e.stage, e.step) for e in events] == [
        (ProgressStage.SAMPLING, 0),
        (ProgressStage.SAMPLING, 7),
        (ProgressStage.SAMPLING, 8),
    ]


def test_errors_and_warnings_are_collected() -> None:
    parser = SdCliOutputParser(total_steps=4)
    parser.feed(SAMPLE)
    parser.feed("[E] cannot inspect model source 'x': No such file --- loader.cpp:25\n")
    assert parser.warnings == ["something odd"]
    assert parser.error_summary() == "cannot inspect model source 'x': No such file"


def test_error_summary_falls_back_to_last_lines() -> None:
    parser = SdCliOutputParser(total_steps=4)
    parser.feed("Segmentation fault\n")
    assert parser.error_summary() == "Segmentation fault"
    assert SdCliOutputParser(total_steps=1).error_summary() == "no output"


def test_parse_version() -> None:
    assert parse_version("stable-diffusion.cpp version unknown, commit 532d2c1\n") == (
        "unknown",
        "532d2c1",
    )
    assert parse_version("stable-diffusion.cpp version master-ff4fdbb, commit FF4FDBB") == (
        "master-ff4fdbb",
        "ff4fdbb",
    )
    assert parse_version("something else") == ("something else", None)
