#!/usr/bin/env python3
"""
text-csv converter  —  handles 2TB+ workloads
Usage:
    python convert.py                        # use config/config.json defaults
    python convert.py --workers 8            # override worker count
    python convert.py --input /mnt/data      # override input dir
    python convert.py --resume               # skip already-done files
    python convert.py --single file.txt      # convert one file
"""

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

from src.worker import worker_convert


# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def resolve_dirs(cfg: dict) -> Tuple[Path, Path, Path, Path]:
    input_dir  = BASE_DIR / cfg.get("input_dir",      "input")
    output_dir = BASE_DIR / cfg.get("output_dir",     "output")
    log_dir    = BASE_DIR / cfg.get("log_dir",        "logs")
    ckpt_dir   = BASE_DIR / cfg.get("checkpoint_dir", "checkpoints")
    for d in (input_dir, output_dir, log_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir, log_dir, ckpt_dir


# ── File discovery ────────────────────────────────────────────────────────────

def discover_files(input_dir: Path, filter_cfg: dict) -> List[Path]:
    exts      = set(filter_cfg.get("extensions", [".txt", ".log", ".dat", ".tsv"]))
    recursive = filter_cfg.get("recursive", True)
    min_size  = filter_cfg.get("min_size_bytes", 0)
    max_size  = filter_cfg.get("max_size_bytes")

    pattern = "**/*" if recursive else "*"
    files = []
    for p in input_dir.glob(pattern):
        if not p.is_file():
            continue
        if exts and p.suffix.lower() not in exts:
            continue
        sz = p.stat().st_size
        if sz < min_size:
            continue
        if max_size and sz > max_size:
            continue
        files.append(p)

    files.sort(key=lambda p: p.stat().st_size, reverse=True)  # big files first
    return files


# ── Job building ──────────────────────────────────────────────────────────────

def build_jobs(
    files: List[Path],
    input_dir: Path,
    output_dir: Path,
    ckpt_dir: Path,
    cfg: dict,
    resume: bool,
) -> List[tuple]:
    jobs = []
    done_marker_suffix = ".done"

    for src in files:
        rel = src.relative_to(input_dir)
        dst = (output_dir / rel).with_suffix(".csv")
        ckpt = ckpt_dir / (str(rel).replace(os.sep, "_") + ".ckpt")
        done_marker = ckpt_dir / (str(rel).replace(os.sep, "_") + done_marker_suffix)

        if resume and done_marker.exists():
            continue  # already completed

        jobs.append((str(src), str(dst), str(ckpt), cfg))

    return jobs


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path) -> logging.Logger:
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"convert_{ts}.log"
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("text-csv")


# ── Progress display ──────────────────────────────────────────────────────────

def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def human_speed(bps: float) -> str:
    return human_bytes(int(bps)) + "/s"


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args, cfg: dict, log: logging.Logger):
    input_dir, output_dir, log_dir, ckpt_dir = resolve_dirs(cfg)

    if args.input:
        input_dir = Path(args.input)
    if args.output:
        output_dir = Path(args.output)

    # Single-file mode
    if args.single:
        src = Path(args.single)
        dst = output_dir / src.with_suffix(".csv").name
        ckpt = ckpt_dir / (src.name + ".ckpt")
        log.info(f"Single file: {src} → {dst}")
        result = worker_convert((str(src), str(dst), str(ckpt), cfg))
        log.info(
            f"Done: {result['rows']:,} rows  |  "
            f"{human_bytes(result['bytes_read'])}  |  "
            f"{result['elapsed']:.1f}s"
            + (f"  |  ERROR: {result['error']}" if result.get("error") else "")
        )
        return

    filter_cfg = cfg.get("file_filter", {})
    files = discover_files(input_dir, filter_cfg)
    log.info(f"Found {len(files):,} files in {input_dir}")

    if not files:
        log.warning("No files found. Drop text files into the 'input' folder and re-run.")
        return

    total_bytes = sum(p.stat().st_size for p in files)
    log.info(f"Total size: {human_bytes(total_bytes)}")

    jobs = build_jobs(files, input_dir, output_dir, ckpt_dir, cfg, resume=args.resume)
    log.info(f"Jobs to run: {len(jobs):,}  ({'resume mode' if args.resume else 'full mode'})")

    if not jobs:
        log.info("Nothing to do (all files already converted). Use --no-resume to reprocess.")
        return

    n_workers = args.workers or cfg.get("workers", 0) or max(1, multiprocessing.cpu_count() - 1)
    n_workers = min(n_workers, len(jobs))
    log.info(f"Workers: {n_workers}")

    t_start = time.perf_counter()
    total_rows = 0
    total_read = 0
    errors = []

    done_suffix = ".done"

    with multiprocessing.Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_convert, jobs), 1):
            src = result["src"]
            elapsed = result["elapsed"]
            rows = result["rows"]
            read = result["bytes_read"]
            err = result.get("error")

            total_rows += rows
            total_read += read

            if err:
                errors.append((src, err))
                log.error(f"[{i}/{len(jobs)}] FAILED {Path(src).name}: {err}")
            else:
                speed = read / elapsed if elapsed > 0 else 0
                log.info(
                    f"[{i}/{len(jobs)}] {Path(src).name}  "
                    f"{rows:,} rows  {human_bytes(read)}  "
                    f"{elapsed:.1f}s  {human_speed(speed)}"
                )
                # Mark as done for resume
                rel = Path(src).relative_to(input_dir)
                done_marker = ckpt_dir / (str(rel).replace(os.sep, "_") + done_suffix)
                done_marker.touch()

            # ETA
            elapsed_total = time.perf_counter() - t_start
            pct = i / len(jobs)
            eta = (elapsed_total / pct - elapsed_total) if pct > 0 else 0
            log.info(
                f"  Progress: {pct*100:.1f}%  "
                f"ETA: {eta/60:.1f} min  "
                f"Total read: {human_bytes(total_read)}"
            )

    total_elapsed = time.perf_counter() - t_start
    avg_speed = total_read / total_elapsed if total_elapsed > 0 else 0

    log.info("=" * 60)
    log.info(f"Completed: {len(jobs) - len(errors):,} files")
    log.info(f"Total rows written: {total_rows:,}")
    log.info(f"Total data read:    {human_bytes(total_read)}")
    log.info(f"Total time:         {total_elapsed/60:.1f} min")
    log.info(f"Average speed:      {human_speed(avg_speed)}")
    if errors:
        log.error(f"Failed files ({len(errors)}):")
        for src, err in errors:
            log.error(f"  {src}: {err}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert text files to CSV at scale")
    parser.add_argument("--config",   default="config/config.json", help="Path to config file")
    parser.add_argument("--input",    help="Override input directory")
    parser.add_argument("--output",   help="Override output directory")
    parser.add_argument("--workers",  type=int, default=0, help="Number of parallel workers (0=auto)")
    parser.add_argument("--resume",   action="store_true", default=True,  help="Skip already-converted files (default: on)")
    parser.add_argument("--no-resume",action="store_true", help="Reprocess all files")
    parser.add_argument("--single",   help="Convert a single file and exit")
    args = parser.parse_args()

    if args.no_resume:
        args.resume = False

    cfg_path = BASE_DIR / args.config
    cfg = load_config(cfg_path)

    log = setup_logging(BASE_DIR / cfg.get("log_dir", "logs"))
    log.info("text-csv  —  starting")
    log.info(f"Config: {cfg_path}")

    run(args, cfg, log)


if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed for Windows PyInstaller builds
    main()
