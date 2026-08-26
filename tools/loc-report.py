#!/usr/bin/env python3
"""Lines-of-code report for the robotiq workspace.

Splits every Rust file four ways — code, doc/comment, blank, and test —
because a single "lines" number hides the thing we actually care about:
how much *logic* exists versus how much of it is exercised.

Test lines are counted separately (everything inside `#[cfg(test)]`), so
the code/test ratio is honest rather than inflated by the tests themselves.

Usage:  python3 tools/loc-report.py [--markdown]
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {"target", ".git", "node_modules", "vendor"}


def classify(path):
    """Return (code, doc, blank, test, total) line counts for one file."""
    code = doc = blank = test = 0
    in_test = False
    test_depth = 0
    in_block_comment = False

    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    for i, raw in enumerate(lines):
        line = raw.strip()

        # Enter a #[cfg(test)] block: the next `mod ... {` starts it.
        if not in_test and re.match(r"#\[cfg\(test\)\]", line):
            in_test = True
            test_depth = 0
            test += 1
            continue

        if in_test:
            test += 1
            test_depth += line.count("{") - line.count("}")
            # Depth returns to 0 only after the mod block has opened and closed.
            if (
                test_depth <= 0
                and "{" in "".join(lines[max(0, i - 1) : i + 1])
                and test_depth == 0
                and "}" in line
            ):
                in_test = False
            continue

        if not line:
            blank += 1
            continue

        if in_block_comment:
            doc += 1
            if "*/" in line:
                in_block_comment = False
            continue

        if line.startswith("/*"):
            doc += 1
            if "*/" not in line:
                in_block_comment = True
            continue

        if line.startswith("//"):
            doc += 1
            continue

        code += 1

    return code, doc, blank, test, len(lines)


def crate_of(rel):
    parts = rel.split(os.sep)
    if parts[0] in ("crates", "firmware") and len(parts) > 1:
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def collect():
    rows = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if not fn.endswith(".rs"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ROOT)
            rows.append((crate_of(rel), rel, *classify(full)))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def main():
    markdown = "--markdown" in sys.argv
    rows = collect()

    crates = {}
    for crate, _rel, code, doc, blank, test, total in rows:
        agg = crates.setdefault(crate, [0, 0, 0, 0, 0, 0])
        agg[0] += code
        agg[1] += doc
        agg[2] += blank
        agg[3] += test
        agg[4] += total
        agg[5] += 1

    bar = "|" if markdown else ""
    sep = " | " if markdown else "  "

    def emit(cells, widths):
        out = sep.join(
            c.rjust(w) if i else c.ljust(w)
            for i, (c, w) in enumerate(zip(cells, widths, strict=True))
        )
        print(f"{bar} {out} {bar}" if markdown else out)

    widths = [34, 7, 7, 7, 7, 7, 7]
    header = ["file", "code", "doc", "blank", "test", "total", "doc%"]

    print("\n## Per-file\n")
    emit(header, widths)
    if markdown:
        print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    else:
        print("-" * (sum(widths) + 2 * len(widths)))

    last = None
    for crate, rel, code, doc, blank, test, total in rows:
        if crate != last and not markdown:
            print(f"\n[{crate}]")
            last = crate
        pct = f"{100 * doc / (code + doc):.0f}%" if (code + doc) else "-"
        name = os.path.basename(rel) if not markdown else rel
        emit(
            [name, str(code), str(doc), str(blank), str(test), str(total), pct], widths
        )

    print("\n## Per-crate\n")
    emit(["crate", "code", "doc", "blank", "test", "total", "files"], widths)
    if markdown:
        print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    else:
        print("-" * (sum(widths) + 2 * len(widths)))

    tot = [0, 0, 0, 0, 0, 0]
    for crate in sorted(crates):
        c = crates[crate]
        emit(
            [crate, str(c[0]), str(c[1]), str(c[2]), str(c[3]), str(c[4]), str(c[5])],
            widths,
        )
        for i in range(6):
            tot[i] += c[i]

    print()
    emit(
        [
            "TOTAL",
            str(tot[0]),
            str(tot[1]),
            str(tot[2]),
            str(tot[3]),
            str(tot[4]),
            str(tot[5]),
        ],
        widths,
    )

    code, doc, test = tot[0], tot[1], tot[3]
    print()
    print(f"code:test ratio     1 : {test / code:.2f}" if code else "")
    print(f"doc:code ratio      1 : {code / doc:.2f}" if doc else "")
    print(
        f"documentation       {100 * doc / (code + doc):.1f}% "
        "of non-blank non-test lines"
    )


if __name__ == "__main__":
    main()
