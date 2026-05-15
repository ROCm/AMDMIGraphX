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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
"""Benchmark `.mxr` files dumped by ``MIGRAPHX_GPU_DUMP_BENCHMARK_MXR``.

For each problem encountered (one or more competing solutions saved as separate
``.mxr`` files), each solution is timed using the same recipe hipBLASLt's
exhaustive tune uses (``time_loop`` in ``src/targets/gpu/time_op.cpp``):

    1. one untimed warmup call,
    2. ``nruns`` iterations of ``bundle`` consecutive ``program.run`` calls,
    3. sort the per-call samples and average the middle 50% (drop top and
       bottom 25%) to produce the "common average" used elsewhere in MIGraphX.

Defaults match the hipBLASLt path (``bundle=4``, ``nruns=40``). The
``MIGRAPHX_BENCHMARKING_BUNDLE`` and ``MIGRAPHX_BENCHMARKING_NRUNS`` env vars
are honored, the same names the C++ runtime checks.

The fastest solution per problem is written to a JSON file in the same shape
``MIGRAPHX_PROBLEM_CACHE`` reads (an array of ``[key, value]`` pairs where
each key is ``{"name": <preop>, "problem": <problem>}`` and the value is the
solution).
"""

import argparse
import json
import os
import re
import sys
import time
from collections import OrderedDict

import migraphx


# ---------------------------------------------------------------------------
# Parser for the `migraphx::to_string(value)` format produced by
# `compile_plan::save_binaries`. The text inside the `@comment` op is shaped
# like:
#
#     "<preop_name> problem={...} solution={...}"
#
# The `{...}` is `migraphx::to_string` of a `value` (see `print_value` in
# `src/value.cpp`). That format renders BOTH arrays and objects with `{}`,
# numeric/identifier atoms as bare tokens, and key/value pairs separated by
# `:`. Strings inside values (e.g. "float_type") are emitted unquoted.
#
# Object vs array is disambiguated by looking at the first element inside a
# `{...}` block: if it is followed by `:`, the block is an object, otherwise
# an array.
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(
    r"\s+"
    r'|"(?:\\.|[^"\\])*"'
    r"|-?(?:nan|inf|infinity)\b"
    r"|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?"
    r"|-?\.\d+(?:[eE][+\-]?\d+)?"
    r"|[A-Za-z_][A-Za-z0-9_.]*"
    r"|::"
    r"|[{}\[\]:,]")

_LITERAL_KEYWORDS = {
    "null": None,
    "true": True,
    "false": True,
    "nan": float("nan"),
    "inf": float("inf"),
    "infinity": float("inf"),
    "-nan": float("nan"),
    "-inf": float("-inf"),
    "-infinity": float("-inf"),
}
_LITERAL_KEYWORDS["false"] = False


def _tokenize(text):
    tokens = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise ValueError(
                f"Could not tokenize at position {pos}: {text[pos:pos+20]!r}")
        tok = m.group(0)
        if not tok.isspace():
            tokens.append(tok)
        pos = m.end()
    return tokens


def _atom_to_python(tok):
    if tok.startswith('"') and tok.endswith('"'):
        return bytes(tok[1:-1], "utf-8").decode("unicode_escape")
    lower = tok.lower()
    if lower in _LITERAL_KEYWORDS:
        return _LITERAL_KEYWORDS[lower]
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass
    return tok


class _TokenStream:
    def __init__(self, tokens):
        self._tokens = tokens
        self._idx = 0

    def peek(self):
        return self._tokens[self._idx] if self._idx < len(self._tokens) else None

    def take(self):
        tok = self.peek()
        self._idx += 1
        return tok

    def expect(self, expected):
        tok = self.take()
        if tok != expected:
            raise ValueError(f"Expected {expected!r}, got {tok!r}")


def _parse_value(stream):
    tok = stream.peek()
    if tok is None:
        raise ValueError("Unexpected end of input")
    if tok == "{":
        return _parse_brace(stream)
    if tok == "[":
        return _parse_bracket(stream)
    return _atom_to_python(stream.take())


def _parse_brace(stream):
    stream.expect("{")
    if stream.peek() == "}":
        stream.take()
        return []
    first = _parse_value(stream)
    if stream.peek() == ":":
        stream.take()
        result = OrderedDict()
        result[str(first)] = _parse_value(stream)
        while stream.peek() == ",":
            stream.take()
            key = _parse_value(stream)
            stream.expect(":")
            result[str(key)] = _parse_value(stream)
        stream.expect("}")
        return result
    items = [first]
    while stream.peek() == ",":
        stream.take()
        items.append(_parse_value(stream))
    stream.expect("}")
    return items


def _parse_bracket(stream):
    stream.expect("[")
    if stream.peek() == "]":
        stream.take()
        return []
    items = [_parse_value(stream)]
    while stream.peek() == ",":
        stream.take()
        items.append(_parse_value(stream))
    stream.expect("]")
    return items


def parse_value_string(text):
    """Parse a ``migraphx::to_string(value)`` rendering into Python objects.

    Two flavors are supported:

    * Structured values that begin with ``{`` or ``[`` are tokenized and
      parsed as objects/arrays of identifiers, numbers, literals (``null``,
      ``true``, ``false``, ``nan``, ``inf``), and quoted strings.
    * Bare atoms, including arbitrary strings containing whitespace or
      punctuation that the structural tokenizer does not handle, are
      returned as-is. ``gpu::mlir_op`` uses a raw rocMLIR tuning key
      (e.g. ``arch+:sramecc+:xnack-\\t304\\t8\\tconv\\t...``) for both
      problem and solution; see ``get_tuning_config`` in
      ``src/targets/gpu/mlir.cpp``.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty value string")
    if stripped[0] not in "{[":
        return _atom_to_python(stripped)
    stream = _TokenStream(_tokenize(stripped))
    result = _parse_value(stream)
    leftover = stream.peek()
    if leftover is not None:
        raise ValueError(f"Unexpected trailing token: {leftover!r}")
    return result


# ---------------------------------------------------------------------------
# Comment text splitter.
#
# Format (see `compile_ops.cpp::compile_plan::save_binaries`):
#   "<preop_name> problem={...} solution={...}"
#
# We split with a tiny brace-depth scanner so nested `{}` inside the problem
# value don't confuse the boundary between problem and solution.
# ---------------------------------------------------------------------------


def parse_comment_text(text):
    """Return ``(preop_name, problem_obj, solution_obj)`` parsed from the text
    of an ``@comment`` instruction inserted by ``save_binaries``.

    The text is shaped as ``"<preop> problem=<value> solution=<value>"``,
    where each ``<value>`` is the output of ``migraphx::to_string`` and may
    be either a ``{...}`` block or a bare atom (e.g. an int solution index)."""
    problem_marker = " problem="
    solution_marker = " solution="
    p_idx = text.find(problem_marker)
    s_idx = text.rfind(solution_marker)
    if p_idx < 0 or s_idx < 0 or s_idx < p_idx:
        raise ValueError(
            f"Comment text does not match the expected format: {text!r}")
    preop_name = text[:p_idx].strip()
    problem_str = text[p_idx + len(problem_marker):s_idx].strip()
    solution_str = text[s_idx + len(solution_marker):].strip()
    if not problem_str or not solution_str:
        raise ValueError(
            f"Empty problem or solution in comment text: {text!r}")
    return (preop_name, parse_value_string(problem_str),
            parse_value_string(solution_str))


# ---------------------------------------------------------------------------
# Metadata extraction from a loaded program.
# ---------------------------------------------------------------------------


def extract_comment_metadata(prog):
    """Walk the program's main module and return the parsed metadata of the
    ``@comment`` instruction that ``save_binaries`` inserts."""
    mm = prog.get_main_module()
    for ins in mm:
        if ins.op().name() == "@comment":
            text = ins.op().values().get("text", "")
            return parse_comment_text(text)
    raise ValueError("No @comment instruction found in program")


# ---------------------------------------------------------------------------
# Benchmarking loop. Mirrors `time_loop` in
# `src/targets/gpu/time_op.cpp` (the routine called by hipBLASLt's
# `tune()` in `src/targets/gpu/hip_gemm_impl.cpp`):
#
#   - 1 untimed warmup call
#   - nruns iterations, each timing a bundle of `bundle` consecutive runs
#   - sort the nruns samples, drop top 25% and bottom 25%, average the rest
#
# The Python bindings don't expose HIP events, so we use `time.perf_counter_ns`
# around each bundle and sync the GPU at the end of every bundle so the timer
# captures the full GPU work for that bundle.
# ---------------------------------------------------------------------------


def _attach_gpu_context(prog):
    """Attach a GPU target/context to a program loaded from an `.mxr` file.

    Programs saved by ``compile_plan::save_binaries`` are produced by
    ``compile_plan::make_program()`` (see ``src/targets/gpu/compile_ops.cpp``),
    which builds a fresh ``program`` with the compiled GPU code objects but
    without ever assigning a target. Their ``targets`` / ``contexts`` arrays
    are empty when serialized, so ``program::from_value`` skips its
    auto-finalize branch and ``prog.run(...)`` immediately fails with
    ``No context available for gpu::code_object``.

    We use ``program.finalize(target)`` -- which mirrors what the C++
    ``time_program`` (src/targets/gpu/time_op.cpp) does internally -- to
    attach the target + context and call ``module::finalize`` without
    running any compile passes. Calling ``program.compile(target)`` instead
    would re-run the full GPU pass list including ``auto_contiguous``,
    which destructively rewrites broadcast strides and corrupts the saved
    ``gpu::code_object_op``'s ``expected_inputs``.
    """
    if not prog.is_compiled():
        prog.finalize(migraphx.get_target("gpu"))


def _build_param_map(prog, seed):
    params = {}
    next_seed = seed
    for name, shape in prog.get_parameter_shapes().items():
        host_arg = migraphx.generate_argument(shape, next_seed)
        params[name] = migraphx.to_gpu(host_arg)
        next_seed += 1
    return params


def _trimmed_mean_ms(samples):
    samples = sorted(samples)
    n = len(samples)
    drop = n // 4
    keep = samples[drop:n - drop] if n - 2 * drop > 0 else samples
    return sum(keep) / len(keep)


def benchmark_program(prog, bundle, nruns, seed=0):
    """Return the trimmed-mean ms-per-call for ``prog``."""
    _attach_gpu_context(prog)
    params = _build_param_map(prog, seed)

    prog.run(params)
    migraphx.gpu_sync()

    samples = []
    for _ in range(nruns):
        start = time.perf_counter_ns()
        for _ in range(bundle):
            prog.run(params)
        migraphx.gpu_sync()
        elapsed_ms = (time.perf_counter_ns() - start) / 1.0e6
        samples.append(elapsed_ms / bundle)

    return _trimmed_mean_ms(samples)


# ---------------------------------------------------------------------------
# Driver: scan a directory of .mxr files, group by (preop, problem), benchmark
# each, pick the fastest, and write a MIGRAPHX_PROBLEM_CACHE-compatible JSON.
# ---------------------------------------------------------------------------


def _problem_key(preop, problem_obj):
    return (preop, json.dumps(problem_obj, sort_keys=True, default=str))


_FILENAME_HASH_RE = re.compile(r"_(\d+)\.mxr$")


def _problem_tag_from_filename(name):
    """Extract the trailing problem-hash suffix that ``compile_plan::save_binaries``
    appends to dumped filenames (``<preop>_<solution_idx>_<problem_hash>.mxr``).
    Used purely as a short, stable label to distinguish multiple winners with
    the same ``preop`` in the console log."""
    m = _FILENAME_HASH_RE.search(os.path.basename(name))
    return m.group(1) if m else None


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(
            f"warning: {name}={raw!r} is not an integer; using {default}",
            file=sys.stderr)
        return default


def _scan_mxr_files(mxr_dir, pattern):
    if not os.path.isdir(mxr_dir):
        raise FileNotFoundError(f"Not a directory: {mxr_dir}")
    regex = re.compile(
        "^" + re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".") +
        "$")
    paths = []
    for entry in sorted(os.listdir(mxr_dir)):
        if regex.match(entry):
            full = os.path.join(mxr_dir, entry)
            if os.path.isfile(full):
                paths.append(full)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=
        "Benchmark .mxr files dumped via MIGRAPHX_GPU_DUMP_BENCHMARK_MXR "
        "and emit a MIGRAPHX_PROBLEM_CACHE-compatible JSON file.")
    parser.add_argument("mxr_dir",
                        help="Directory containing the dumped .mxr files.")
    parser.add_argument(
        "-o",
        "--output",
        default="problem_cache.json",
        help="Path for the problem-cache JSON file (default: %(default)s).")
    parser.add_argument(
        "--bundle",
        type=int,
        default=None,
        help=
        "Bundle size: number of consecutive program.run() calls per timing "
        "sample. Defaults to MIGRAPHX_BENCHMARKING_BUNDLE if set, else 4 "
        "(the value hipBLASLt's exhaustive tune uses).")
    parser.add_argument(
        "--nruns",
        type=int,
        default=None,
        help=
        "Number of timing samples to collect per .mxr file. Defaults to "
        "MIGRAPHX_BENCHMARKING_NRUNS if set, else 40.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=
        "Seed used by migraphx.generate_argument when filling parameters "
        "with random data (default: %(default)s).")
    parser.add_argument(
        "--pattern",
        default="*.mxr",
        help="Filename glob to match inside mxr_dir (default: %(default)s).")
    args = parser.parse_args(argv)

    bundle = args.bundle if args.bundle is not None else _env_int(
        "MIGRAPHX_BENCHMARKING_BUNDLE", 4)
    nruns = args.nruns if args.nruns is not None else _env_int(
        "MIGRAPHX_BENCHMARKING_NRUNS", 40)
    if bundle < 1 or nruns < 1:
        parser.error("--bundle and --nruns must both be >= 1")

    files = _scan_mxr_files(args.mxr_dir, args.pattern)
    if not files:
        parser.error(
            f"No files matching {args.pattern!r} found in {args.mxr_dir}")

    print(
        f"Benchmarking {len(files)} file(s) from {args.mxr_dir} "
        f"(bundle={bundle}, nruns={nruns})",
        flush=True)

    groups = OrderedDict()
    for path in files:
        rel = os.path.relpath(path, args.mxr_dir)
        try:
            prog = migraphx.load(path)
            preop, problem_obj, solution_obj = extract_comment_metadata(prog)
        except Exception as exc:
            print(f"  [skip] {rel}: {exc}", file=sys.stderr)
            continue

        try:
            avg_ms = benchmark_program(prog,
                                       bundle=bundle,
                                       nruns=nruns,
                                       seed=args.seed)
        except Exception as exc:
            print(f"  [skip] {rel}: benchmark failed: {exc}", file=sys.stderr)
            continue

        key = _problem_key(preop, problem_obj)
        bucket = groups.setdefault(
            key, {
                "preop": preop,
                "problem": problem_obj,
                "problem_tag": _problem_tag_from_filename(rel),
                "candidates": []
            })
        bucket["candidates"].append({
            "file": rel,
            "solution": solution_obj,
            "ms": avg_ms,
        })
        print(f"  {rel}: {preop}  -> {avg_ms:.6f} ms/call", flush=True)

    if not groups:
        print("error: no benchmarks completed successfully", file=sys.stderr)
        return 1

    cache_entries = []
    for bucket in groups.values():
        candidates = bucket["candidates"]
        winner = min(candidates, key=lambda c: c["ms"])
        cache_entries.append([{
            "name": bucket["preop"],
            "problem": bucket["problem"]
        }, winner["solution"]])
        tag = bucket["problem_tag"] or "?"
        print(
            f"  {bucket['preop']} (problem={tag}): best = {winner['file']} "
            f"({winner['ms']:.6f} ms/call) over {len(candidates)} candidate(s)",
            flush=True)

    with open(args.output, "w") as f:
        json.dump(cache_entries, f, indent=4)
        f.write("\n")
    print(
        f"Wrote {len(cache_entries)} entry(ies) to {args.output}", flush=True)
    print(
        f"Use it via: MIGRAPHX_PROBLEM_CACHE={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
