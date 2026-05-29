#!/usr/bin/env python3
"""
Summarize MIGraphX ONNX Model Zoo test logs.
Reads *.log files from fp32/ and fp16/ under a results directory,
counts passes/failures, extracts a short failure message, and writes:
  - summary.json
  - summary.md
Extras:
  - --previous: compare to previous run's summary.json (pass-rate delta)
  - --write-index: generate index.md with a clickable table of models
  - --artifact-url: best-effort link target used in index rows (run page)
  - --step-summary: write a compact overview into $GITHUB_STEP_SUMMARY
Usage:
  python3 scripts/summarize_onnx_logs.py --results <RESULTS_DIR> \
      [--out-json <PATH>] [--out-md <PATH>] \
      [--previous <PATH>] [--write-index <PATH>] \
      [--artifact-url <URL>] [--step-summary <PATH>]
"""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path

FAIL_PAT = re.compile(
    r"(Traceback \(most recent call last\)|\bERROR\b|AssertionError|Segmentation fault|^error:)",
    re.I | re.M,
)
TBLOCK_PAT = re.compile(r"Traceback \(most recent call last\):([\s\S]*?)(?:\n\s*\n|\Z)")


def looks_failed(text: str) -> bool:
    return bool(FAIL_PAT.search(text))


def failure_message(text: str) -> str:
    # last traceback block, last non-empty line
    blocks = list(TBLOCK_PAT.finditer(text))
    if blocks:
        for line in reversed(blocks[-1].group(1).strip().splitlines()):
            line = line.strip()
            if line:
                return line
    # fallback: last interesting line
    for line in reversed([l.strip() for l in text.splitlines() if l.strip()]):
        if re.search(r"(error|exception|failed|segmentation fault|assert)", line, re.I):
            return line
    return "failed (see log)"


def summarize(results_dir: Path) -> dict:
    precs = ("fp32", "fp16")
    summary = {"totals": {"pass": 0, "fail": 0}, "regressions": {}}

    for prec in precs:
        d = results_dir / prec
        reg = {"passed": [], "failed": []}
        if d.is_dir():
            files = sorted(p for p in d.glob("*.log"))
            for p in files:
                model = p.stem
                try:
                    txt = p.read_text(errors="ignore")
                except Exception:
                    txt = ""
                if looks_failed(txt):
                    reg["failed"].append({"model": model, "message": failure_message(txt)})
                    summary["totals"]["fail"] += 1
                else:
                    reg["passed"].append(model)
                    summary["totals"]["pass"] += 1
        summary["regressions"][prec] = reg
    return summary


def write_outputs(summary: dict, out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(summary, indent=2))

    lines = [
        "## Totals",
        f"- PASS: {summary['totals']['pass']}",
        f"- FAIL: {summary['totals']['fail']}",
        "",
    ]
    for prec in ("fp32", "fp16"):
        reg = summary["regressions"][prec]
        lines.append(f"## {prec.upper()}")
        lines.append(f"**Passed ({len(reg['passed'])})**")
        if reg["passed"]:
            lines.extend([f"- {m}" for m in reg["passed"]])
        else:
            lines.append("- none")
        lines.append("")
        lines.append(f"**Failed ({len(reg['failed'])})**")
        if reg["failed"]:
            lines.extend([f"- {it['model']}: `{it['message']}`" for it in reg["failed"]])
        else:
            lines.append("- none")
        lines.append("")
    out_md.write_text("\n".join(lines))


def build_index_md(summary: dict, artifact_url: str | None) -> str:
    """Return an index.md string with a model table and best-effort links."""
    # Collect models and statuses
    models = set()
    status: dict[str, dict[str, str]] = {"fp32": {}, "fp16": {}}
    for prec in ("fp32", "fp16"):
        for m in summary["regressions"][prec]["passed"]:
            models.add(m)
            status[prec][m] = "pass"
        for it in summary["regressions"][prec]["failed"]:
            models.add(it["model"])
            status[prec][it["model"]] = "fail"

    def cell(prec: str, model: str) -> str:
        st = status[prec].get(model)
        if not st:
            return ""
        emoji = "✅" if st == "pass" else "❌"
        rel = f"{prec}/{model}.log"
        # If artifact_url is provided, link to the run page and show the relative path below.
        # (Direct deep-links to a file in an artifact page are not guaranteed.)
        if artifact_url:
            return f"[{emoji}]({artifact_url})<br/><sub>{rel}</sub>"
        else:
            # When opening index.md *inside* the artifact, relative links work.
            return f"[{emoji}]({rel})"

    rows = ["| Model | FP32 | FP16 |", "|---|:---:|:---:|"]
    for model in sorted(models):
        rows.append(f"| {model} | {cell('fp32', model)} | {cell('fp16', model)} |")
    return "\n".join(rows)


def write_step_summary(summary: dict, prev: dict | None, index_hint: str, step_summary_path: Path) -> None:
    cur_total = summary["totals"]["pass"] + summary["totals"]["fail"]
    cur_rate = (summary["totals"]["pass"] / cur_total) * 100 if cur_total else 0.0
    if prev:
        prev_total = prev["totals"]["pass"] + prev["totals"]["fail"]
        prev_rate = (prev["totals"]["pass"] / prev_total) * 100 if prev_total else 0.0
        delta = cur_rate - prev_rate
        comp = f"**Pass rate:** {cur_rate:.1f}% (Δ {delta:+.1f} pts vs. previous {prev_rate:.1f}%)"
    else:
        comp = f"**Pass rate:** {cur_rate:.1f}% (no previous run found)"

    failed = []
    for prec in ("fp32", "fp16"):
        failed.extend([f"{prec}:{it['model']} — {it['message'][:80]}" for it in summary["regressions"][prec]["failed"]])
    failed_block = "\n".join([f"- {l}" for l in failed[:20]]) or "- none"

    text = [
        "# MIGraphX ONNX Model Zoo — Summary",
        comp,
        "",
        f"Artifacts: {index_hint}",
        "",
        "## Top failures (first 20)",
        failed_block,
    ]
    # Write (overwrites prior content for this step)
    step_summary_path.write_text("\n".join(text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Results directory containing fp32/fp16 log folders")
    ap.add_argument("--out-json", default=None, help="Path to write summary.json (default: <results>/summary.json)")
    ap.add_argument("--out-md", default=None, help="Path to write summary.md (default: <results>/summary.md)")
    ap.add_argument("--previous", default=None, help="Path to previous summary.json for comparison")
    ap.add_argument("--artifact-url", default=None, help="Run or artifact URL to use for links in index.md (best-effort)")
    ap.add_argument("--write-index", default=None, help="Write a model table index.md here (optional)")
    ap.add_argument("--step-summary", default=None, help="Write a compact overview to this path (e.g., $GITHUB_STEP_SUMMARY)")
    args = ap.parse_args()

    results_dir = Path(args.results).expanduser().resolve()
    if not results_dir.exists():
        print(f"ERROR: results dir not found: {results_dir}")
        return 2

    summary = summarize(results_dir)

    out_json = Path(args.out_json) if args.out_json else results_dir / "summary.json"
    out_md = Path(args.out_md) if args.out_md else results_dir / "summary.md"
    write_outputs(summary, out_json, out_md)

    prev = None
    if args.previous and Path(args.previous).exists():
        try:
            prev = json.loads(Path(args.previous).read_text())
        except Exception:
            prev = None

    if args.write_index:
        idx = build_index_md(summary, args.artifact_url)
        Path(args.write_index).write_text(idx)

    if args.step_summary:
        hint = args.artifact_url or "(open run → Artifacts → logs artifact → index.md)"
        write_step_summary(summary, prev, hint, Path(args.step_summary))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
