#!/usr/bin/env python3
"""
Analyze CK vs MLIR split-KV benchmark CSVs: M,N,K,O patterns vs verdict.

Expects CSV columns: num_splits, batch, nhead, M, N, K, O, ck_time_ms,
mlir_time_ms, faster, delta_ms, speedup_pct, (optional extra run columns).

Verdict bands (default neutral window):
  neutral:    speedup_pct in [-neutral_pct, +neutral_pct]
  ck_win:     speedup_pct > +neutral_pct   (CK materially faster)
  mlir_win:   speedup_pct < -neutral_pct  (MLIR materially faster)

Marginal BY M/N/K/O, BY (K,O) pairs, and BY O/K ratio bucket are emitted as CSV
tables with the same columns. CK speedup columns use raw speedup_pct for ck_win
rows; MLIR columns use abs(speedup_pct) for mlir_win rows.

O/K ratio CSV lists fine bands in fixed order (mirrored K/O and O/K ranges plus
``O=K``). Two additional tables cross each of those buckets with ``M`` and with
``N`` (``criteria`` like ``2<K/O<=4,M=32``).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = frozenset(
    {"M", "N", "K", "O", "faster", "speedup_pct", "num_splits"}
)


def verdict(speedup_pct: float, neutral_pct: float) -> str:
    if -neutral_pct <= speedup_pct <= neutral_pct:
        return "neutral"
    if speedup_pct > neutral_pct:
        return "ck_win"
    return "mlir_win"


def percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def abs_speedups(rows: list[dict[str, Any]], which: str) -> list[float]:
    return sorted(abs(x["speedup_pct"]) for x in rows if x["verdict"] == which)


def load_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path}: empty or no header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for lineno, row in enumerate(reader, start=2):
            try:
                m = int(row["M"])
                n = int(row["N"])
                k = int(row["K"])
                o = int(row["O"])
                sp = float(row["speedup_pct"])
                faster = row["faster"].strip()
                num_splits = int(row["num_splits"])
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(f"{path}:{lineno}: bad row: {e}") from e
            rows.append(
                {
                    "path": path.name,
                    "num_splits": num_splits,
                    "M": m,
                    "N": n,
                    "K": k,
                    "O": o,
                    "faster": faster,
                    "speedup_pct": sp,
                    "line": lineno,
                }
            )
    return rows


def discover_csvs(directory: Path, pattern: str) -> list[Path]:
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} under {directory}")
    return paths


def ratio_bucket(k: int, o: int) -> str | None:
    """
    Classify (K, O) by O/K vs K/O so coarse bands mirror across O=K.

    When O < K (O/K < 1), K/O > 1; we split like the O > K side:
      K/O > 4, then 2 < K/O <= 4, then 1 < K/O <= 2.
    That covers O/K in (0, 1) in three reciprocal bands (down to arbitrarily
    small O/K as K/O grows).

    When O = K: O=K. When O > K: 1 < O/K <= 2, 2 < O/K <= 4, O/K > 4.
    """
    if k == 0:
        return None
    r = o / k
    if r < 1:
        # O < K  =>  K/O > 1 (integer-safe thresholds)
        if k > 4 * o:
            return "K/O>4"
        if k > 2 * o:
            return "2<K/O<=4"
        # o < k <= 2*o  =>  1 < K/O <= 2
        return "1<K/O<=2"
    if r == 1:
        return "O=K"
    if r <= 2:
        return "1<O/K<=2"
    if r <= 4:
        return "2<O/K<=4"
    return "O/K>4"


MARGINAL_CSV_FIELDS = (
    "criteria",
    "neutral",
    "ck_faster",
    "mlir_faster",
    "ck_min_speedup",
    "ck_max_speedup",
    "ck_median_speedup",
    "ck_average_speedup",
    "mlir_min_speedup",
    "mlir_max_speedup",
    "mlir_median_speedup",
    "mlir_average_speedup",
)


def _speedup_stats_fields(
    speedups: list[float], prefix: str
) -> dict[str, str]:
    """min/max/median/mean as string columns; empty list -> all empty strings."""
    empty = {
        f"{prefix}_min_speedup": "",
        f"{prefix}_max_speedup": "",
        f"{prefix}_median_speedup": "",
        f"{prefix}_average_speedup": "",
    }
    if not speedups:
        return empty
    s = sorted(speedups)
    mean = sum(speedups) / len(speedups)
    med = percentile(s, 50)
    assert med is not None
    return {
        f"{prefix}_min_speedup": f"{s[0]:.4f}",
        f"{prefix}_max_speedup": f"{s[-1]:.4f}",
        f"{prefix}_median_speedup": f"{med:.4f}",
        f"{prefix}_average_speedup": f"{mean:.4f}",
    }


def write_verdict_rollup_csv(
    section_title: str,
    buckets: list[tuple[str, list[dict[str, Any]]]],
) -> None:
    """
    One CSV table: criteria, verdict counts, CK/MLIR speedup stats per bucket.
    MLIR speedup columns use abs(speedup_pct).
    """
    print(f"\n=== {section_title} ===")
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=list(MARGINAL_CSV_FIELDS),
        lineterminator="\n",
    )
    writer.writeheader()
    for criteria, xs in buckets:
        cc = Counter(z["verdict"] for z in xs)
        n0, n1, n2 = cc["neutral"], cc["ck_win"], cc["mlir_win"]
        ck_sp = [z["speedup_pct"] for z in xs if z["verdict"] == "ck_win"]
        ml_sp = [
            abs(z["speedup_pct"])
            for z in xs
            if z["verdict"] == "mlir_win"
        ]
        row: dict[str, Any] = {
            "criteria": criteria,
            "neutral": n0,
            "ck_faster": n1,
            "mlir_faster": n2,
        }
        row.update(_speedup_stats_fields(ck_sp, "ck"))
        row.update(_speedup_stats_fields(ml_sp, "mlir"))
        writer.writerow(row)


def write_o_k_bucket_cross_dim_csv(
    dim: str,
    bucket_series: list[tuple[str, list[dict[str, Any]]]],
) -> None:
    """
    For each O/K bucket (same order as the main O/K table), emit one CSV row per
    distinct ``dim`` value (M or N) seen inside that bucket. ``criteria`` looks
    like ``2<K/O<=4,M=32``.
    """
    cross_buckets: list[tuple[str, list[dict[str, Any]]]] = []
    for bucket_name, xs in bucket_series:
        if not xs:
            continue
        by_dim: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for x in xs:
            by_dim[x[dim]].append(x)
        for key in sorted(by_dim.keys()):
            cross_buckets.append((f"{bucket_name},{dim}={key}", by_dim[key]))
    write_verdict_rollup_csv(f"BY O/K bucket × {dim} (CSV)", cross_buckets)


def run_report(rows: list[dict[str, Any]], neutral_pct: float) -> None:
    for x in rows:
        x["verdict"] = verdict(x["speedup_pct"], neutral_pct)

    bad_faster = [x for x in rows if x["faster"] not in ("ck", "mlir")]
    sign_mismatch: list[dict[str, Any]] = []
    for x in rows:
        if x["faster"] == "ck" and x["speedup_pct"] < 0:
            sign_mismatch.append(x)
        if x["faster"] == "mlir" and x["speedup_pct"] > 0:
            sign_mismatch.append(x)

    # print("=== FILES ===")
    # by_file = Counter(x["path"] for x in rows)
    # for name in sorted(by_file):
    #     print(f"  {name}: {by_file[name]}")
    # print(f"  TOTAL: {len(rows)}")

    # print("\n=== CONSISTENCY ===")
    # print(f"  invalid faster (not ck/mlir): {len(bad_faster)}")
    # print(f"  faster vs speedup_pct sign mismatch: {len(sign_mismatch)}")
    # for x in sign_mismatch[:20]:
    #     print(
    #         f"    {x['path']} L{x['line']}: faster={x['faster']!r} "
    #         f"speedup_pct={x['speedup_pct']}"
    #     )

    vc = Counter(x["verdict"] for x in rows)
    print(f"\n=== VERDICT (±{neutral_pct:g}% neutral) — merged ===")
    for k in ("neutral", "ck_win", "mlir_win"):
        n = vc[k]
        pct = 100.0 * n / len(rows) if rows else 0.0
        print(f"  {k}: {n} ({pct:.1f}%)")

    non_neu = [x for x in rows if x["verdict"] != "neutral"]
    nn = len(non_neu)
    if nn:
        ck_nn = sum(1 for x in non_neu if x["verdict"] == "ck_win")
        ml_nn = sum(1 for x in non_neu if x["verdict"] == "mlir_win")
        print(f"\n=== EXCLUDING NEUTRAL ({nn} rows) ===")
        print(
            f"  CK material win: {ck_nn} ({100.0 * ck_nn / nn:.1f}% of non-neutral)"
        )
        print(
            f"  MLIR material win: {ml_nn} ({100.0 * ml_nn / nn:.1f}% of non-neutral)"
        )

    # print("\n=== MAGNITUDE |speedup_pct| (material wins only) ===")
    # for label, subset in (
    #     ("all rows", rows),
    # ):
    #     ck = [x for x in subset if x["verdict"] == "ck_win"]
    #     ml = [x for x in subset if x["verdict"] == "mlir_win"]
    #     print(f"  [{label}]")
    #     a_ck = abs_speedups(subset, "ck_win")
    #     a_ml = abs_speedups(subset, "mlir_win")
    #     if a_ck:
    #         print(
    #             f"    CK wins: n={len(a_ck)} median|sp|={percentile(a_ck, 50):.2f}% "
    #             f"p90|sp|={percentile(a_ck, 90):.2f}%"
    #         )
    #     if a_ml:
    #         print(
    #             f"    MLIR wins: n={len(a_ml)} median|sp|={percentile(a_ml, 50):.2f}% "
    #             f"p90|sp|={percentile(a_ml, 90):.2f}%"
    #         )

    def marginal_csv(dim: str) -> None:
        by: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for x in rows:
            by[x[dim]].append(x)
        buckets = [(f"{dim}={key}", by[key]) for key in sorted(by.keys())]
        write_verdict_rollup_csv(f"BY {dim} (CSV)", buckets)

    for d in ("M", "N", "K", "O"):
        marginal_csv(d)

    cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for x in rows:
        cells[(x["K"], x["O"])].append(x)
    ko_buckets = [
        (f"K={k} O={o}", cells[(k, o)])
        for k, o in sorted(cells.keys())
    ]
    write_verdict_rollup_csv("(K,O) (CSV)", ko_buckets)

    # print("\n=== BY num_splits ===")
    # for ns in sorted({x["num_splits"] for x in rows}):
    #     sub = [x for x in rows if x["num_splits"] == ns]
    #     c = Counter(z["verdict"] for z in sub)
    #     t = len(sub)
    #     print(
    #         f"  num_splits={ns}: n={t} neutral={100.0 * c['neutral'] / t:.1f}% "
    #         f"ck_win={100.0 * c['ck_win'] / t:.1f}% "
    #         f"mlir_win={100.0 * c['mlir_win'] / t:.1f}%"
    #     )

    rb: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for x in rows:
        b = ratio_bucket(x["K"], x["O"])
        if b:
            rb[b].append(x)
    order = [
        "K/O>4",
        "2<K/O<=4",
        "1<K/O<=2",
        "O=K",
        "1<O/K<=2",
        "2<O/K<=4",
        "O/K>4",
    ]
    ok_bucket_series = [(b, rb[b]) for b in order if b in rb]
    write_verdict_rollup_csv("BY O/K ratio bucket (CSV)", ok_bucket_series)
    write_o_k_bucket_cross_dim_csv("M", ok_bucket_series)
    write_o_k_bucket_cross_dim_csv("N", ok_bucket_series)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "directory",
        type=Path,
        help="Directory containing benchmark CSV files",
    )
    p.add_argument(
        "--glob",
        default="*.csv",
        help="Glob pattern under directory (default: *.csv)",
    )
    p.add_argument(
        "--neutral-pct",
        type=float,
        default=5.0,
        help="Half-width of neutral band in speedup_pct (default: 5)",
    )
    args = p.parse_args()
    directory = args.directory.expanduser().resolve()
    if not directory.is_dir():
        print(f"Not a directory: {directory}", file=sys.stderr)
        return 2
    paths = discover_csvs(directory, args.glob)
    all_rows: list[dict[str, Any]] = []
    for path in paths:
        all_rows.extend(load_csv(path))
    run_report(all_rows, neutral_pct=args.neutral_pct)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
