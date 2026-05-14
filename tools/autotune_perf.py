#!/usr/bin/env python3
#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################

"""Run migraphx-driver perf under curated MIGraphX environment-variable knobs."""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from typing import Iterable

TOTAL_TIME_MS = re.compile(r"Total time:\s*([0-9]+\.?[0-9]*)\s*ms")

KNOBS: tuple[tuple[str, str, str], ...] = (
    ("NHWC layout", "MIGRAPHX_ENABLE_NHWC", "1"),
    ("Disable hipBLASLt", "MIGRAPHX_DISABLE_HIPBLASLT", "1"),
    ("Enable CK GEMM", "MIGRAPHX_ENABLE_CK", "1"),
    ("Disable MLIR", "MIGRAPHX_DISABLE_MLIR", "1"),
    ("Conv->dot rewrite", "MIGRAPHX_ENABLE_REWRITE_DOT", "1"),
)


def parse_total_time_ms(output: str) -> float | None:
    found = TOTAL_TIME_MS.search(output)
    if found is None:
        return None
    return float(found.group(1))


def resolve_driver(explicit: str | None) -> str:
    if explicit:
        return explicit
    env_path = os.environ.get("MIGRAPHX_DRIVER")
    if env_path:
        return env_path
    which = shutil.which("migraphx-driver")
    if which:
        return which
    print(
        "error: migraphx-driver not found. Pass --driver PATH or set MIGRAPHX_DRIVER.",
        file=sys.stderr,
    )
    sys.exit(1)


def warn_confounded_env(knobs: Iterable[tuple[str, str, str]]) -> None:
    for _, name, _ in knobs:
        if name in os.environ:
            print(
                f"warning: {name} is set in the environment; "
                "results may be confounded.",
                file=sys.stderr,
            )


def run_perf(
    driver: str,
    perf_argv: list[str],
    env_name: str | None,
    env_value: str | None,
) -> float | None:
    env = os.environ.copy()
    if env_name is not None and env_value is not None:
        env[env_name] = env_value
    proc = subprocess.run(
        [driver, *perf_argv],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    text = proc.stdout or ""
    return parse_total_time_ms(text)


def write_config(
    path: str,
    model_file: str,
    baseline_ms: float,
    env_name: str,
    env_value: str,
    winner_ms: float,
) -> None:
    pct = (winner_ms - baseline_ms) / baseline_ms * 100.0
    pct_prefix = "+" if pct >= 0 else ""
    lines = (
        f"# Autotune config for {model_file}",
        f"# Baseline: {baseline_ms} ms",
        f"# Winner:   {winner_ms} ms ({pct_prefix}{pct}%)",
        "#",
        "# Source this file before running migraphx-driver / your application.",
        f"export {env_name}={env_value}",
        "",
    )
    with open(path, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(lines))


def default_config_path(perf_argv: list[str], explicit: str | None) -> str:
    if explicit:
        return explicit
    if len(perf_argv) >= 2:
        return perf_argv[1] + ".tune"
    return "migraphx_perf.tune"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep a curated set of MIGraphX environment variables (one at a time), "
            "run migraphx-driver perf for each, and report the fastest configuration."
        ),
        epilog=(
            "Example: %(prog)s --driver ./build/bin/migraphx-driver perf model.onnx "
            "--iterations 50 --cpu --log-level error"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--driver",
        metavar="PATH",
        help="migraphx-driver binary (default: MIGRAPHX_DRIVER or PATH)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Write winning exports here (default: <model>.tune)",
    )
    args, perf_argv = parser.parse_known_args()
    if not perf_argv:
        parser.error(
            "missing perf invocation e.g. perf <model.onnx> [--iterations N] [...]; "
            "pass driver flags after script options."
        )

    perf_argv = list(perf_argv)
    if perf_argv[0] != "perf":
        perf_argv.insert(0, "perf")

    driver = resolve_driver(args.driver)
    warn_confounded_env(KNOBS)

    rows: list[tuple[str, str | None, str | None]] = [("baseline", None, None)]
    rows.extend((lab, nam, val) for lab, nam, val in KNOBS)

    model_display = perf_argv[1] if len(perf_argv) >= 2 else "(unknown)"
    print(f"Autotune: {len(rows)} configurations on {model_display}")

    label_width = max(len(r[0]) for r in rows) + 2
    times: list[float | None] = []
    for index, (label, env_name, env_value) in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] {label} ... ", end="", flush=True)
        t_ms = run_perf(driver, perf_argv, env_name, env_value)
        times.append(t_ms)
        if t_ms is None:
            print("failed")
        else:
            print(f"{t_ms} ms")

    baseline = times[0]
    if baseline is None:
        print("error: baseline failed; cannot rank configurations.", file=sys.stderr)
        sys.exit(1)

    win_index = min(
        range(len(times)),
        key=lambda i: times[i] if times[i] is not None else math.inf,
    )

    print("\nResults:")
    for i, ((label, env_name, env_value), t_ms) in enumerate(zip(rows, times)):
        gap = label_width - len(label)
        gap = max(gap, 1)
        line = f"  {label}{' ' * gap}"
        if t_ms is None:
            print(f"{line}failed")
            continue
        line += f"{t_ms} ms"
        if env_name is not None:
            pct = (t_ms - baseline) / baseline * 100.0
            pct_prefix = "+" if pct >= 0 else ""
            line += f"  {pct_prefix}{pct}%"
        if i == win_index and env_name is not None:
            line += "  <-- best"
        print(line)

    _, win_env, win_val = rows[win_index]
    win_t = times[win_index]
    if win_env is None or win_val is None or win_t is None:
        print("\nBaseline is best; no config written.")
        return

    out_path = default_config_path(perf_argv, args.output)
    write_config(out_path, model_display, baseline, win_env, win_val, win_t)
    print(f"\nConfig written: {out_path}")


if __name__ == "__main__":
    main()
