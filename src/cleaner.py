"""
Professional data cleaner for converted CSV files.
Operations: whitespace normalization, null standardization, case normalization,
            deduplication, type inference, IQR-based outlier detection, profiling.
"""

import re
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


_NULL_DEFAULTS = {
    "", "null", "none", "n/a", "na", "-", "nan", "nil",
    "NULL", "N/A", "None", "NaN", "NA", "missing",
}


def _is_outside(val: str, bound: Tuple[float, float]) -> bool:
    try:
        f = float(val)
        return f < bound[0] or f > bound[1]
    except ValueError:
        return False


def _percentile(sorted_data: List[float], pct: float) -> float:
    if not sorted_data:
        return 0.0
    idx = (len(sorted_data) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


class DataCleaner:
    def __init__(self, cfg: dict):
        c = cfg.get("cleaning", {})
        self.strip_ws = c.get("strip_whitespace", True)
        self.normalize_case = c.get("normalize_case")       # "lower" | "upper" | "title" | None
        null_vals = c.get("null_values", list(_NULL_DEFAULTS))
        self.null_values = {str(v).lower() for v in null_vals}
        self.null_replacement = c.get("null_replacement", "")
        self.strip_chars = c.get("strip_chars", [";"])
        self.repair_merged = c.get("repair_merged_cells", True)
        self.remove_duplicates = c.get("remove_duplicates", True)
        self.duplicate_subset = c.get("duplicate_subset")  # list of col names, or None = all cols
        self.outlier_enabled = c.get("outlier_detection", True)
        self.outlier_multiplier = float(c.get("outlier_iqr_multiplier", 1.5))
        self.outlier_action = c.get("outlier_action", "flag")  # flag | remove | cap
        self.report_enabled = c.get("report", True)

    # ── Streaming API (for large files) ──────────────────────────────────────

    def init_streaming(self, header: Optional[List[str]], n_cols: int):
        """Call once before streaming rows through process_row()."""
        self._s_header = header
        self._s_n_cols = n_cols
        self._s_col_names = header[:] if header else [f"col_{i}" for i in range(n_cols)]
        self._s_seen: set = set()
        self._s_col_types: List[str] = ["unknown"] * n_cols
        self._s_reservoir: List[List[float]] = [[] for _ in range(n_cols)]
        self._s_reservoir_max = 50_000          # samples per column for IQR estimate
        self._s_stats = {
            "rows_in": 0, "rows_out": 0,
            "duplicates_removed": 0, "nulls_replaced": 0,
            "whitespace_stripped": 0, "outliers_flagged": 0, "outliers_removed": 0,
        }

    def process_row(self, row: List[str]) -> Optional[List[str]]:
        """Clean + repair + dedup a single row. Returns cleaned row or None."""
        self._s_stats["rows_in"] += 1
        n = self._s_n_cols

        # cell-level cleaning
        new_row: List[str] = []
        for val in row:
            original = val
            if self.strip_ws:
                val = val.strip()
                if val != original:
                    self._s_stats["whitespace_stripped"] += 1
            if self.strip_chars:
                val = re.sub('[' + re.escape(''.join(self.strip_chars)) + ']', '', val).strip()
            if val.lower() in self.null_values:
                val = self.null_replacement
                self._s_stats["nulls_replaced"] += 1
            elif val and self.normalize_case:
                if self.normalize_case == "lower":
                    val = val.lower()
                elif self.normalize_case == "upper":
                    val = val.upper()
                elif self.normalize_case == "title":
                    val = val.title()
            new_row.append(val)

        # strip trailing empty
        while new_row and new_row[-1] == self.null_replacement:
            new_row.pop()

        # repair merged cells
        if self.repair_merged:
            new_row = self._repair_row(new_row, n)
            if new_row is None:
                return None

        # align to n_cols
        while len(new_row) < n:
            new_row.append(self.null_replacement)
        new_row = new_row[:n]

        # deduplication using hash (memory-efficient)
        if self.remove_duplicates:
            col_names = self._s_col_names
            if self.duplicate_subset:
                idx = [col_names.index(c) for c in self.duplicate_subset if c in col_names]
                if not idx:
                    idx = list(range(n))
            else:
                idx = list(range(n))
            key = hash(tuple(new_row[i] for i in idx if i < len(new_row)))
            if key in self._s_seen:
                self._s_stats["duplicates_removed"] += 1
                return None
            self._s_seen.add(key)

        # update type inference + reservoir for IQR
        for i, val in enumerate(new_row):
            if i >= n:
                break
            if self._s_col_types[i] == "string" or not val:
                continue
            try:
                fval = float(val)
                if self._s_col_types[i] == "unknown":
                    self._s_col_types[i] = "numeric"
                if len(self._s_reservoir[i]) < self._s_reservoir_max:
                    self._s_reservoir[i].append(fval)
            except ValueError:
                self._s_col_types[i] = "string"

        self._s_stats["rows_out"] += 1
        return new_row

    def compute_streaming_bounds(self) -> List[Optional[Tuple[float, float]]]:
        """Compute IQR outlier bounds from reservoir samples collected during process_row()."""
        bounds: List[Optional[Tuple[float, float]]] = []
        for i, col_type in enumerate(self._s_col_types):
            if col_type == "numeric" and self._s_reservoir[i]:
                vals = sorted(self._s_reservoir[i])
                q1 = _percentile(vals, 0.25)
                q3 = _percentile(vals, 0.75)
                iqr = q3 - q1
                bounds.append((q1 - self.outlier_multiplier * iqr,
                                q3 + self.outlier_multiplier * iqr))
            else:
                bounds.append(None)
        return bounds

    def apply_outlier_row(
        self, row: List[str], bounds: List[Optional[Tuple[float, float]]]
    ) -> Optional[List[str]]:
        """Apply outlier action to a single row. Returns row (possibly modified) or None."""
        is_outlier = any(
            bound is not None and i < len(row) and _is_outside(row[i], bound)
            for i, bound in enumerate(bounds)
        )
        if is_outlier:
            if self.outlier_action == "remove":
                self._s_stats["outliers_removed"] += 1
                return None
            elif self.outlier_action == "flag":
                self._s_stats["outliers_flagged"] += 1
                return row + ["1"]
            elif self.outlier_action == "cap":
                row = list(row)
                for i, bound in enumerate(bounds):
                    if bound is None or i >= len(row):
                        continue
                    try:
                        row[i] = str(round(max(bound[0], min(bound[1], float(row[i]))), 6))
                    except ValueError:
                        pass
                return row
        if self.outlier_action == "flag":
            return row + ["0"]
        return row

    # ── Row repair ────────────────────────────────────────────────────────────

    def _repair_row(self, row: List[str], n_cols: int) -> Optional[List[str]]:
        non_empty = [v for v in row if v]

        if len(non_empty) == 1:
            val = non_empty[0]
            # name-age-city  e.g. mike-23-bkk
            m = re.match(r'^([a-zA-Z]\w*)-(\d+)-(\w+)$', val)
            if m:
                result = list(m.groups())
                while len(result) < n_cols:
                    result.append(self.null_replacement)
                return result[:n_cols]
            # age:city  e.g. 23:bangkok — no name, drop row
            if re.match(r'^\d+:\w+$', val):
                return None

        # digit+alpha merge  e.g. 20bangkok  →  20, bangkok
        new_row: List[str] = []
        carry: Optional[str] = None
        for val in row:
            m = re.match(r'^(\d+)([a-zA-Z].*)$', val)
            if m:
                new_row.append(m.group(1))
                carry = m.group(2)
            elif carry is not None and val == self.null_replacement:
                new_row.append(carry)
                carry = None
            else:
                if carry is not None:
                    new_row.append(carry)
                    carry = None
                new_row.append(val)
        if carry:
            new_row.append(carry)

        while len(new_row) < n_cols:
            new_row.append(self.null_replacement)
        return new_row[:n_cols]

    def clean(
        self,
        header: Optional[List[str]],
        rows: List[List[str]],
    ) -> Tuple[Optional[List[str]], List[List[str]], dict]:
        """
        Clean in-memory rows. Returns (final_header, cleaned_rows, report).
        header may be None for headerless files.
        """
        # Strip trailing empty header columns
        if header:
            trimmed_header = list(header)
            while trimmed_header and trimmed_header[-1].strip() == "":
                trimmed_header.pop()
            header = trimmed_header

        n_cols = len(header) if header else (max((len(r) for r in rows), default=0))
        col_names = header[:] if header else [f"col_{i}" for i in range(n_cols)]

        report: dict = {
            "total_rows_in": len(rows),
            "total_rows_out": 0,
            "columns": n_cols,
            "column_names": col_names,
            "whitespace_stripped": 0,
            "nulls_replaced": 0,
            "duplicates_removed": 0,
            "outliers_flagged": 0,
            "outliers_removed": 0,
            "column_profiles": {},
        }

        # ── Step 1: cell-level cleaning ───────────────────────────────────────
        cleaned: List[List[str]] = []
        for row in rows:
            new_row: List[str] = []
            for val in row:
                original = val
                if self.strip_ws:
                    val = val.strip()
                    if val != original:
                        report["whitespace_stripped"] += 1
                if self.strip_chars:
                    val = re.sub('[' + re.escape(''.join(self.strip_chars)) + ']', '', val).strip()
                if val.lower() in self.null_values:
                    val = self.null_replacement
                    report["nulls_replaced"] += 1
                elif val and self.normalize_case:
                    if self.normalize_case == "lower":
                        val = val.lower()
                    elif self.normalize_case == "upper":
                        val = val.upper()
                    elif self.normalize_case == "title":
                        val = val.title()
                new_row.append(val)
            # Strip trailing empty cells
            while new_row and new_row[-1] == self.null_replacement:
                new_row.pop()
            # Align to expected column count
            while len(new_row) < n_cols:
                new_row.append(self.null_replacement)
            cleaned.append(new_row[:n_cols])

        # ── Step 1b: row repair ───────────────────────────────────────────────
        if self.repair_merged:
            repaired = []
            for row in cleaned:
                fixed = self._repair_row(row, n_cols)
                if fixed is not None:
                    repaired.append(fixed)
            cleaned = repaired

        # ── Step 2: deduplication ─────────────────────────────────────────────
        if self.remove_duplicates:
            # resolve subset column indices (ignore id/auto-increment cols if specified)
            if self.duplicate_subset:
                subset_idx = [
                    col_names.index(c) for c in self.duplicate_subset if c in col_names
                ]
                if not subset_idx:   # named columns not found → fall back to all columns
                    subset_idx = list(range(n_cols))
            else:
                subset_idx = list(range(n_cols))

            seen: set = set()
            deduped: List[List[str]] = []
            for row in cleaned:
                key = tuple(row[i] for i in subset_idx if i < len(row))
                if key in seen:
                    report["duplicates_removed"] += 1
                else:
                    seen.add(key)
                    deduped.append(row)
            cleaned = deduped

        # ── Step 3: type inference + column profiling ─────────────────────────
        col_types: List[str] = []
        col_numeric: List[List[float]] = [[] for _ in col_names]

        for i, col in enumerate(col_names):
            null_count = 0
            numeric_count = 0
            unique_vals: set = set()

            for row in cleaned:
                val = row[i] if i < len(row) else ""
                if val == self.null_replacement or val == "":
                    null_count += 1
                    continue
                unique_vals.add(val)
                try:
                    col_numeric[i].append(float(val))
                    numeric_count += 1
                except ValueError:
                    pass

            non_null = len(cleaned) - null_count
            is_numeric = non_null > 0 and numeric_count == non_null
            col_types.append("numeric" if is_numeric else "string")

            profile: dict = {
                "type": "numeric" if is_numeric else "string",
                "null_count": null_count,
                "null_pct": round(null_count / len(cleaned) * 100, 2) if cleaned else 0.0,
                "unique_count": len(unique_vals),
            }
            if is_numeric and col_numeric[i]:
                vals = sorted(col_numeric[i])
                mean = sum(vals) / len(vals)
                profile.update({
                    "min": vals[0],
                    "max": vals[-1],
                    "mean": round(mean, 4),
                    "median": round(_percentile(vals, 0.5), 4),
                    "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
                    "q1": round(_percentile(vals, 0.25), 4),
                    "q3": round(_percentile(vals, 0.75), 4),
                })
            report["column_profiles"][col] = profile

        # ── Step 4: outlier detection ─────────────────────────────────────────
        final_header = header[:] if header else None
        if self.outlier_enabled:
            bounds: List[Optional[Tuple[float, float]]] = []
            for i, col in enumerate(col_names):
                if col_types[i] == "numeric" and col_numeric[i]:
                    vals = sorted(col_numeric[i])
                    q1 = _percentile(vals, 0.25)
                    q3 = _percentile(vals, 0.75)
                    iqr = q3 - q1
                    bounds.append((
                        q1 - self.outlier_multiplier * iqr,
                        q3 + self.outlier_multiplier * iqr,
                    ))
                else:
                    bounds.append(None)

            if self.outlier_action == "flag":
                if final_header is not None:
                    final_header = final_header + ["_outlier"]

            result: List[List[str]] = []
            for row in cleaned:
                is_outlier = False
                for i, bound in enumerate(bounds):
                    if bound is None or i >= len(row):
                        continue
                    try:
                        fval = float(row[i])
                        if fval < bound[0] or fval > bound[1]:
                            is_outlier = True
                            break
                    except ValueError:
                        pass

                if is_outlier:
                    if self.outlier_action == "remove":
                        report["outliers_removed"] += 1
                        continue
                    elif self.outlier_action == "flag":
                        report["outliers_flagged"] += 1
                        result.append(row + ["1"])
                    elif self.outlier_action == "cap":
                        row = list(row)
                        for i, bound in enumerate(bounds):
                            if bound is None or i >= len(row):
                                continue
                            try:
                                fval = float(row[i])
                                row[i] = str(round(max(bound[0], min(bound[1], fval)), 6))
                            except ValueError:
                                pass
                        result.append(row)
                    else:
                        result.append(row)
                else:
                    if self.outlier_action == "flag":
                        result.append(row + ["0"])
                    else:
                        result.append(row)

            cleaned = result

        report["total_rows_out"] = len(cleaned)
        return final_header, cleaned, report
