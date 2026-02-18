"""CLI argument parsers and file I/O helpers (CSV, LaTeX tables)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


def parse_int_list(s: str) -> List[int]:
    """Parse a comma-separated string of integers."""
    s = (s or "").strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    """Parse a comma-separated string of floats."""
    s = (s or "").strip()
    if not s:
        return []
    return [float(x) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> List[str]:
    """Parse a comma-separated string of strings."""
    return [x.strip() for x in s.split(",") if x.strip()]


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str] | None = None):
    """Write a list of dicts as CSV. Infers fieldnames from first row if not given."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_tex_table(path: Path, rows: List[Dict], columns: List[str], header: List[str]):
    """Write a minimal LaTeX tabular from a list of dicts.

    Parameters
    ----------
    path : Path
        Output .tex file.
    rows : list of dict
        Each dict maps column keys to values.
    columns : list of str
        Keys to extract from each row (in order).
    header : list of str
        Column header strings for the LaTeX table.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    col_spec = " ".join(["c"] * len(columns))
    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        " & ".join(header) + r" \\",
        r"\hline",
    ]
    for r in rows:
        vals = [str(r.get(c, "")) for c in columns]
        lines.append(" & ".join(vals) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines), encoding="utf-8")
