#!/usr/bin/env python3
"""
text-csv data cleaner — post-processing step for converted CSVs.
Usage:
    python clean.py                            # clean all CSVs in output/
    python clean.py --input output/sample.csv  # clean a single file
    python clean.py --in-place                 # overwrite originals
    python clean.py --outlier-action remove    # remove outliers (flag|remove|cap)
    python clean.py --no-header                # treat first row as data, not header
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import tempfile
import time
from pathlib import Path

csv.field_size_limit(sys.maxsize)   # remove 128 KB cell limit

from src.cleaner import DataCleaner


BASE_DIR = Path(__file__).parent


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"clean_{ts}.log"
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("text-csv-cleaner")


def _dst_path(src: Path, output_dir: Path, in_place: bool) -> Path:
    if in_place:
        return src
    return output_dir / src.name


STREAM_THRESHOLD_MB = 50   # files larger than this use streaming (no full RAM load)


def clean_file(
    src: Path,
    dst: Path,
    cleaner: DataCleaner,
    has_header: bool,
    log: logging.Logger,
) -> dict:
    size_mb = src.stat().st_size / (1024 * 1024)
    if size_mb > STREAM_THRESHOLD_MB:
        bloom_mb = round(-cleaner.bloom_capacity * math.log(cleaner.bloom_error_rate)
                         / (math.log(2) ** 2) / 8 / 1024 / 1024, 1)
        log.info(f"  Large file ({size_mb:.0f} MB) — streaming mode  "
                 f"| Bloom filter dedup: {bloom_mb} MB RAM")
        return _clean_streaming(src, dst, cleaner, has_header)
    return _clean_inmemory(src, dst, cleaner, has_header)


def _clean_inmemory(src, dst, cleaner, has_header):
    t0 = time.perf_counter()
    try:
        with open(src, newline="", encoding="utf-8") as f:
            all_rows = list(csv.reader(f))
        if not all_rows:
            return {"error": "empty file", "elapsed": 0.0}
        header = all_rows[0] if has_header else None
        rows   = all_rows[1:] if has_header else all_rows
        final_header, cleaned_rows, report = cleaner.clean(header, rows)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if final_header is not None:
                writer.writerow(final_header)
            writer.writerows(cleaned_rows)
    except Exception as exc:
        return {"error": str(exc), "elapsed": round(time.perf_counter() - t0, 3)}
    report["elapsed"] = round(time.perf_counter() - t0, 3)
    report["src"] = str(src)
    report["dst"] = str(dst)
    return report


def _clean_streaming(src, dst, cleaner, has_header):
    """Two-pass streaming cleaner for large files — constant memory usage."""
    t0 = time.perf_counter()
    try:
        # ── Read header ───────────────────────────────────────────────────────
        with open(src, newline="", encoding="utf-8") as f:
            first = next(csv.reader(f), None)
        if first is None:
            return {"error": "empty file", "elapsed": 0.0}

        if has_header:
            header = list(first)
            while header and header[-1].strip() == "":
                header.pop()
            n_cols = len(header)
        else:
            header = None
            n_cols = len(first)

        cleaner.init_streaming(header, n_cols)

        # ── Pass 1: clean + dedup → temp file ────────────────────────────────
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv", prefix="textcsv_clean_")
        os.close(tmp_fd)

        with open(src, newline="", encoding="utf-8") as fin, \
             open(tmp_path, "w", newline="", encoding="utf-8") as ftmp:
            reader = csv.reader(fin)
            writer = csv.writer(ftmp)
            if has_header:
                next(reader)          # skip header row
            for row in reader:
                cleaned = cleaner.process_row(row)
                if cleaned is not None:
                    writer.writerow(cleaned)

        s = cleaner._s_stats

        # ── Pass 2: outlier filter → final output ─────────────────────────────
        bounds = cleaner.compute_streaming_bounds() if cleaner.outlier_enabled else None
        add_outlier_col = (cleaner.outlier_enabled and cleaner.outlier_action == "flag")
        final_header = (header + ["_outlier"]) if (header and add_outlier_col) else header

        dst.parent.mkdir(parents=True, exist_ok=True)
        rows_out = 0
        with open(tmp_path, newline="", encoding="utf-8") as ftmp, \
             open(dst, "w", newline="", encoding="utf-8") as fout:
            reader = csv.reader(ftmp)
            writer = csv.writer(fout)
            if final_header:
                writer.writerow(final_header)
            for row in reader:
                if bounds:
                    row = cleaner.apply_outlier_row(row, bounds)
                    if row is None:
                        continue
                elif add_outlier_col:
                    row = row + ["0"]
                writer.writerow(row)
                rows_out += 1

        os.unlink(tmp_path)

    except Exception as exc:
        if "tmp_path" in dir() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return {"error": str(exc), "elapsed": round(time.perf_counter() - t0, 3)}

    return {
        "total_rows_in":      s["rows_in"],
        "total_rows_out":     rows_out,
        "columns":            n_cols,
        "column_names":       header or [f"col_{i}" for i in range(n_cols)],
        "whitespace_stripped": s["whitespace_stripped"],
        "nulls_replaced":     s["nulls_replaced"],
        "duplicates_removed": s["duplicates_removed"],
        "outliers_flagged":   s["outliers_flagged"],
        "outliers_removed":   s["outliers_removed"],
        "column_profiles":    {},
        "elapsed":            round(time.perf_counter() - t0, 3),
        "src":                str(src),
        "dst":                str(dst),
    }


def _log_report(report: dict, log: logging.Logger):
    log.info(
        f"  Rows : {report['total_rows_in']:>6,} in  ->  {report['total_rows_out']:>6,} out"
    )
    log.info(
        f"  Dupes removed : {report['duplicates_removed']:,}  |  "
        f"Nulls replaced : {report['nulls_replaced']:,}  |  "
        f"Whitespace stripped : {report['whitespace_stripped']:,}"
    )
    log.info(
        f"  Outliers flagged : {report['outliers_flagged']:,}  |  "
        f"Outliers removed : {report['outliers_removed']:,}  |  "
        f"Elapsed : {report['elapsed']:.3f}s"
    )
    log.info("  Column profiles:")
    for col, p in report.get("column_profiles", {}).items():
        if p["type"] == "numeric":
            log.info(
                f"    [{col}]  numeric  "
                f"min={p.get('min')}  max={p.get('max')}  "
                f"mean={p.get('mean')}  std={p.get('std')}  "
                f"nulls={p['null_count']}({p['null_pct']}%)"
            )
        else:
            log.info(
                f"    [{col}]  string   "
                f"unique={p['unique_count']}  nulls={p['null_count']}({p['null_pct']}%)"
            )


def main():
    parser = argparse.ArgumentParser(description="Professional CSV data cleaner")
    parser.add_argument("--config", default="config/config.json")
    parser.add_argument("--input",  help="Input CSV file or directory (default: output/)")
    parser.add_argument("--output-dir", help="Output directory (default: same as input dir)")
    parser.add_argument("--in-place", action="store_true",
                        help="Overwrite input files instead of writing *_clean.csv")
    parser.add_argument("--no-header", action="store_true",
                        help="Treat first row as data (no header row)")
    parser.add_argument("--outlier-action", choices=["flag", "remove", "cap"],
                        help="Override outlier_action from config")
    parser.add_argument("--no-outlier", action="store_true",
                        help="Disable outlier detection entirely")
    args = parser.parse_args()

    cfg_path = BASE_DIR / args.config
    cfg = load_config(cfg_path)

    cleaning_cfg = cfg.setdefault("cleaning", {})
    if args.outlier_action:
        cleaning_cfg["outlier_action"] = args.outlier_action
    if args.no_outlier:
        cleaning_cfg["outlier_detection"] = False

    log_dir = BASE_DIR / cfg.get("log_dir", "logs")
    log = setup_logging(log_dir)
    log.info("text-csv cleaner — starting")
    log.info(f"Config : {cfg_path}")

    input_path = (
        Path(args.input) if args.input
        else BASE_DIR / cfg.get("output_dir", "output")
    )

    if input_path.is_file():
        csv_files = [input_path]
        default_out = BASE_DIR / cfg.get("clean_output_dir", "output_clean")
    else:
        csv_files = sorted(input_path.glob("**/*.csv"))
        if not args.in_place:
            csv_files = [f for f in csv_files if not f.stem.endswith("_clean")]
        default_out = BASE_DIR / cfg.get("clean_output_dir", "output_clean")

    output_dir = Path(args.output_dir) if args.output_dir else default_out

    if not csv_files:
        log.warning(f"No CSV files found in {input_path}")
        return

    log.info(f"Files to clean : {len(csv_files)}")
    has_header = not args.no_header
    cleaner = DataCleaner(cfg)
    all_reports = []

    for src in csv_files:
        dst = _dst_path(src, output_dir, args.in_place)
        log.info(f"Cleaning : {src.name}  ->  {dst.name}")

        report = clean_file(src, dst, cleaner, has_header, log)

        if report.get("error"):
            log.error(f"  ERROR: {report['error']}")
        else:
            _log_report(report, log)
            if cleaning_cfg.get("report", True):
                report_path = log_dir / f"{src.stem}_clean_report.json"
                with open(report_path, "w", encoding="utf-8") as rf:
                    json.dump(report, rf, indent=2, default=str)
                log.info(f"  Report saved : {report_path.name}")

        all_reports.append(report)

    ok = sum(1 for r in all_reports if not r.get("error"))
    log.info("=" * 60)
    log.info(f"Completed : {ok} / {len(csv_files)} files cleaned successfully")


if __name__ == "__main__":
    main()
