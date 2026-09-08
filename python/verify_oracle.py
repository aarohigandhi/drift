"""Check the Java formats against exact rational arithmetic, code by code.

Reads the tables written by ``./gradlew dumpTables`` and recomputes every row from
``drift.minifloat``, which shares nothing with the Java implementation but the
specification. Three things get checked for each format:

  values   the value of every bit pattern
  product  the code that the exact product of every ordered pair rounds to
  sum      the same for the exact sum

For the eight bit and six bit formats every pair is enumerated, so "no
disagreements" covers the entire arithmetic of the format rather than a sample of
it. FP16 and BF16 have 65,536 codes each, which would be 4.3 billion pairs; their
values are still checked exhaustively and their arithmetic on a fixed sample.

Exits non-zero on any disagreement, so CI fails loudly rather than quietly.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import time
from fractions import Fraction
from pathlib import Path

from drift import minifloat as mf


def double_from_hex(h: str) -> float:
    return struct.unpack(">d", bytes.fromhex(h.rjust(16, "0")))[0]


def as_value(x: float) -> mf.Value:
    """A Java-side double turned into the exact value it stands for."""
    if x != x:
        return mf.NAN
    if x == float("inf"):
        return mf.POS_INF
    if x == float("-inf"):
        return mf.NEG_INF
    if x == 0.0:
        # copysign is the only way to see the sign of a zero from Python.
        return mf.NEG_ZERO if math.copysign(1.0, x) < 0 else Fraction(0)
    return Fraction(x)  # exact: every double is a dyadic rational


def read_rows(path: Path):
    with path.open() as fh:
        header = fh.readline()
        fh.readline()  # column names
        for line in fh:
            line = line.strip()
            if line:
                yield line.split(",")
    return header


def check_values(fmt: mf.Fmt, path: Path) -> list[str]:
    """Every bit pattern decodes to the same number on both sides."""
    problems = []
    seen = 0
    for row in read_rows(path):
        code = int(row[0])
        java_value = as_value(double_from_hex(row[1]))
        ours = fmt.decode(code)
        if java_value != ours:
            problems.append(f"{fmt.name} value code={code}: java={java_value} exact={ours}")
        seen += 1
    if seen != fmt.codes:
        problems.append(f"{fmt.name}: expected {fmt.codes} codes, table had {seen}")
    return problems


def check_pairs(fmt: mf.Fmt, path: Path) -> tuple[list[str], int]:
    """
    The product and the sum of two representable values are computed exactly here,
    as rationals, and then rounded once. Java computed them in double arithmetic,
    which is also exact for these widths, and rounded once. The codes must match.
    """
    problems = []
    checked = 0
    for row in read_rows(path):
        a, b = int(row[0]), int(row[1])
        java_prod, java_sum = int(row[2]), int(row[3])
        av, bv = fmt.decode(a), fmt.decode(b)

        ours_prod = fmt.encode(mf.mul(fmt, av, bv))
        if ours_prod != java_prod:
            problems.append(
                f"{fmt.name} product a={a} b={b}: java={java_prod} exact={ours_prod}"
            )
        ours_sum = fmt.encode(mf.add(fmt, av, bv))
        if ours_sum != java_sum:
            problems.append(f"{fmt.name} sum a={a} b={b}: java={java_sum} exact={ours_sum}")
        checked += 1
        if len(problems) > 20:
            problems.append("... stopping after 20")
            break
    return problems, checked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="../results/oracle", type=Path)
    args = ap.parse_args()

    all_problems: list[str] = []
    print(f"{'format':>6}  {'codes':>7}  {'pairs checked':>14}  result")
    print("-" * 48)

    for fmt in mf.ALL:
        vpath = args.tables / f"values_{fmt.name}.csv"
        ppath = args.tables / f"pairs_{fmt.name}.csv"
        if not vpath.exists():
            print(f"{fmt.name:>6}  missing tables, run ./gradlew dumpTables")
            return 2

        print(f"{fmt.name:>6}  {fmt.codes:>7,}  ", end="", flush=True)
        started = time.time()
        problems = check_values(fmt, vpath)
        pair_problems, checked = check_pairs(fmt, ppath)
        problems += pair_problems
        all_problems += problems

        status = "ok" if not problems else f"{len(problems)} DISAGREEMENTS"
        print(f"{checked:>14,}  {status}  ({time.time() - started:.1f}s)")

    print()
    if all_problems:
        for p in all_problems[:40]:
            print("  " + p)
        print(f"\n{len(all_problems)} disagreements")
        return 1

    print("every code and every pair agrees with exact rational arithmetic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
