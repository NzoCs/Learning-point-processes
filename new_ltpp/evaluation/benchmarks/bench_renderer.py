"""Bench Renderer

Reads benchmark JSON results and produces LaTeX tables for inclusion in the
paper. This file provides a small CLI and a `BenchRenderer` class.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


class BenchRenderer:
    """Load benchmark result JSONs and render a LaTeX table."""

    def __init__(self, files: Iterable[str]):
        self.files = list(files)
        self.results: List[Dict[str, Any]] = []

    def load(self) -> None:
        """Load JSON files into memory."""
        for fpath in self.files:
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Benchmark result file not found: {fpath}")
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                data.setdefault("_source_file", os.path.basename(fpath))
                data.setdefault(
                    "model_name",
                    data.get("benchmark_name") or os.path.splitext(os.path.basename(fpath))[0],
                )
                self.results.append(data)

    def collect_metrics(self) -> List[str]:
        keys = set()
        for r in self.results:
            metrics = r.get("metrics") or {}
            keys.update(metrics.keys())
        return sorted(keys)

    @staticmethod
    def _format_value(v: Any, fmt: str = "{:.3f}") -> str:
        try:
            if v is None:
                return "-"
            if isinstance(v, (int, float, np.floating, np.integer)):
                if np.isnan(v):
                    return "-"
                return fmt.format(float(v))
            if isinstance(v, list):
                return ";".join(map(str, v))
            return str(v)
        except Exception:
            return str(v)

    def render_latex_table(
        self,
        metrics: Optional[List[str]] = None,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        bold_best: bool = True,
        float_fmt: str = "{:.3f}",
    ) -> str:
        if not self.results:
            raise RuntimeError("No results loaded. Call load() first.")

        chosen = metrics or self.collect_metrics()
        model_names = [r.get("model_name", r.get("_source_file", f"model{i}")) for i, r in enumerate(self.results)]

        table_values: Dict[str, List[Any]] = {m: [] for m in chosen}
        for m in chosen:
            for r in self.results:
                table_values[m].append(r.get("metrics", {}).get(m, None))

        best_idx: Dict[str, int] = {}
        if bold_best:
            for m, vals in table_values.items():
                numeric = []
                for v in vals:
                    try:
                        numeric.append(float(v) if v is not None else np.nan)
                    except Exception:
                        numeric.append(np.nan)
                if np.all(np.isnan(numeric)):
                    best_idx[m] = -1
                else:
                    best_idx[m] = int(np.nanargmax(numeric))

        col_spec = "l" + "r" * len(chosen)

        lines: List[str] = []
        lines.append("\\begin{table}[ht]")
        lines.append("\\centering")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        # Header row
        header_cells = ["Model"] + [m.replace("_", " ") for m in chosen]
        # use a Python string with escaped backslashes for LaTeX row end
        lines.append(" & ".join(header_cells) + " \\\\")
        lines.append("\\midrule")

        # Rows
        for col_idx, r in enumerate(self.results):
            row_cells = [model_names[col_idx]]
            for m in chosen:
                val = r.get("metrics", {}).get(m, None)
                s = self._format_value(val, fmt=float_fmt)
                if bold_best and best_idx.get(m, -1) == col_idx:
                    s = f"\\textbf{{{s}}}"
                row_cells.append(s)
            lines.append(" & ".join(row_cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{{label}}}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    def save_latex(self, tex: str, out_file: str) -> None:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(tex)


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render LaTeX table from benchmark JSON result files.")
    p.add_argument("--input", "-i", nargs="+", required=True, help="Input JSON files")
    p.add_argument("--metrics", "-m", nargs="*", default=None, help="Metrics to include in the table.")
    p.add_argument("--out-file", "-o", required=True, help="Output .tex file path")
    p.add_argument("--caption", "-c", default=None, help="Caption for the table")
    p.add_argument("--label", "-l", default=None, help="Label for the table")
    p.add_argument("--no-bold", dest="bold", action="store_false", help="Disable bolding of best values per metric")
    p.add_argument("--fmt", default="{:.3f}", help="Float format string, e.g. '{:.2f}'")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_cli()
    args = parser.parse_args(argv)

    renderer = BenchRenderer(args.input)
    renderer.load()
    tex = renderer.render_latex_table(metrics=args.metrics, caption=args.caption, label=args.label, bold_best=args.bold, float_fmt=args.fmt)
    renderer.save_latex(tex, args.out_file)
    print(f"Wrote LaTeX table to: {args.out_file}")


if __name__ == "__main__":
    main()
"""Bench Renderer

This module provides a BenchRenderer class that reads JSON benchmark result
files produced by the benchmarking suite and renders a LaTeX table suitable
for inclusion in the paper. It supports loading multiple result files (from
different models and/or datasets), selecting a subset of metrics, formatting
numbers, and writing a `.tex` table file.

Usage (from project root):
    python -m new_ltpp.evaluation.benchmarks.bench_renderer \
        --input results/modelA_results.json results/modelB_results.json \
        --metrics time_rmse_mean type_accuracy_mean macro_f1score_mean \
        --out-file artifacts/tables/bench_table.tex

The renderer expects JSON files in the format produced by `Benchmark._save_results`.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

class BenchRenderer:
    """Load benchmark result JSONs and render a LaTeX table.

    Responsibilities:
    - Load one or more JSON result files.
    - Normalize metric names and collect per-model values.
    - Optionally compute relative improvements or bold best results.
    - Produce a LaTeX tabular environment as a .tex file.
    """

    def __init__(self, files: Iterable[str]):
        self.files = list(files)
        self.results: List[Dict[str, Any]] = []

    def load(self) -> None:
        """Load JSON files into memory.

        Files that cannot be read will raise an informative exception.
        """
        for fpath in self.files:
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Benchmark result file not found: {fpath}")
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Attach source path for reference
                data.setdefault("_source_file", os.path.basename(fpath))
                # Infer model name if not present
                data.setdefault(
                    "model_name",
                    data.get("benchmark_name") or os.path.splitext(os.path.basename(fpath))[0],
                )
                self.results.append(data)

    def collect_metrics(self) -> List[str]:
        """Collect sorted set of metric keys across loaded results.

        Returns a list of metric keys (as they appear in `results['metrics']`).
        """
        keys = set()
        for r in self.results:
            metrics = r.get("metrics") or {}
            keys.update(metrics.keys())
        # Sort metrics for stable ordering
        return sorted(keys)

    @staticmethod
    def _format_value(v: Any, fmt: str = "{:.3f}") -> str:
        """Format numeric metric or fallback to string."""
        try:
            if v is None:
                return "-"
            if isinstance(v, (int, float, np.floating, np.integer)):
                if np.isnan(v):
                    return "-"
                return fmt.format(float(v))
            # For lists (e.g., confusion matrices) try to summarize
            if isinstance(v, list):
                return ";".join(map(str, v))
            return str(v)
        except Exception:
            return str(v)

    def render_latex_table(
        self,
        metrics: Optional[List[str]] = None,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        bold_best: bool = True,
        float_fmt: str = "{:.3f}",
    ) -> str:
        """Render a LaTeX tabular string for the selected metrics.

        Args:
            metrics: Ordered list of metric keys to include. If None, include all.
            caption: Optional LaTeX caption.
            label: Optional LaTeX label.
            bold_best: If True, bold the best value per metric (for higher-is-better metrics
                       we pick the maximum; this function does not attempt to detect direction,
                       so the caller should arrange metrics accordingly).
            float_fmt: Format string for floats.

        Returns:
            A string containing a complete LaTeX table environment.
        """
        if not self.results:
            raise RuntimeError("No results loaded. Call load() first.")

        all_metrics = self.collect_metrics()
        chosen = metrics or all_metrics

        # Header: model names
        model_names = [r.get("model_name", r.get("_source_file", f"model{i}")) for i, r in enumerate(self.results)]

        # Gather values for table cells
        table_values: Dict[str, List[Any]] = {m: [] for m in chosen}
        for m in chosen:
            for r in self.results:
                val = r.get("metrics", {}).get(m, None)
                table_values[m].append(val)

        # Determine best per metric if requested (we treat NaN or None as worse)
        best_idx: Dict[str, int] = {}
        if bold_best:
            for m, vals in table_values.items():
                numeric = []
                for v in vals:
                    try:
                        if v is None:
                            numeric.append(np.nan)
                        else:
                            numeric.append(float(v))
                    except Exception:
                        numeric.append(np.nan)
                # Best is argmax; NaNs ignored by nanargmax if not all NaN
                if np.all(np.isnan(numeric)):
                    best_idx[m] = -1
                else:
                    best_idx[m] = int(np.nanargmax(numeric))

        # Build LaTeX table
        ncols = 1 + len(chosen)  # first column = model name
        col_spec = "l" + "r" * len(chosen)

        lines: List[str] = []
        lines.append("\\begin{table}[ht]")
        lines.append("\\centering")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        # Header row
        header_cells = ["Model"] + [m.replace("_", " ") for m in chosen]
        lines.append(" & ".join(header_cells) + " \\\\")
        lines.append("\\midrule")

        # Rows: one per model
        for col_idx, r in enumerate(self.results):
            row_cells = [model_names[col_idx]]
            for m in chosen:
                val = r.get("metrics", {}).get(m, None)
                s = self._format_value(val, fmt=float_fmt)
                if bold_best and best_idx.get(m, -1) == col_idx:
                    s = f"\\textbf{{{s}}}"
                row_cells.append(s)
            lines.append(" & ".join(row_cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{{label}}}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    def save_latex(self, tex: str, out_file: str) -> None:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(tex)