"""
Single-file converter: streams text → CSV in chunks to stay memory-efficient.
Designed to handle files of any size (tested against multi-GB files).
"""

import csv
import gzip
import io
import os
import time
from pathlib import Path
from typing import Optional

from src.parser import TextParser


QUOTING_MAP = {
    "minimal": csv.QUOTE_MINIMAL,
    "all": csv.QUOTE_ALL,
    "nonnumeric": csv.QUOTE_NONNUMERIC,
    "none": csv.QUOTE_NONE,
}


def convert_file(
    src_path: Path,
    dst_path: Path,
    cfg: dict,
    checkpoint_path: Optional[Path] = None,
) -> dict:
    """
    Convert a single text file to CSV.
    Returns a stats dict: {rows, bytes_read, elapsed, skipped_lines, error}.
    Supports resuming via checkpoint_path (stores last completed line index).
    """
    parse_cfg = cfg.get("parsing", {})
    out_cfg = cfg.get("output", {})
    perf_cfg = cfg.get("performance", {})

    chunk_size = perf_cfg.get("read_chunk_lines", 50000)
    write_buf = perf_cfg.get("write_buffer_lines", 10000)
    compress = out_cfg.get("compress", False)
    csv_delim = out_cfg.get("csv_delimiter", ",")
    quoting = QUOTING_MAP.get(out_cfg.get("quoting", "minimal"), csv.QUOTE_MINIMAL)
    inc_fname = out_cfg.get("include_filename_column", False)
    inc_lnum = out_cfg.get("include_linenumber_column", False)
    encoding = parse_cfg.get("encoding", "utf-8")
    enc_errors = parse_cfg.get("encoding_errors", "replace")

    resume_line = 0
    if checkpoint_path and checkpoint_path.exists():
        try:
            resume_line = int(checkpoint_path.read_text().strip())
        except Exception:
            resume_line = 0

    parser = TextParser(parse_cfg)
    stats = {"rows": 0, "bytes_read": 0, "elapsed": 0.0, "skipped_lines": 0, "error": None}
    t0 = time.perf_counter()

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    open_fn = gzip.open if compress else open
    dst_suffix = dst_path.with_suffix(".csv.gz") if compress else dst_path

    try:
        with open(src_path, "r", encoding=encoding, errors=enc_errors) as src_f, \
             open_fn(str(dst_suffix), "wt", newline="", encoding="utf-8") as dst_f:

            writer = csv.writer(dst_f, delimiter=csv_delim, quoting=quoting)
            fname = src_path.name
            raw_lnum = 0
            write_buffer = []

            while True:
                raw_chunk = []
                for _ in range(chunk_size):
                    line = src_f.readline()
                    if not line:
                        break
                    raw_lnum += 1
                    raw_chunk.append(line)
                    stats["bytes_read"] += len(line.encode(encoding, errors="replace"))

                if not raw_chunk:
                    break

                # Skip already-processed lines when resuming
                if raw_lnum - len(raw_chunk) < resume_line:
                    skip_count = resume_line - (raw_lnum - len(raw_chunk))
                    raw_chunk = raw_chunk[skip_count:]
                    if not raw_chunk:
                        continue

                rows = list(parser.parse_lines(raw_chunk))

                for row in rows:
                    out_row = []
                    if inc_fname:
                        out_row.append(fname)
                    if inc_lnum:
                        out_row.append(stats["rows"] + 1)
                    out_row.extend(row)
                    write_buffer.append(out_row)
                    stats["rows"] += 1

                    if len(write_buffer) >= write_buf:
                        writer.writerows(write_buffer)
                        write_buffer.clear()

                if write_buffer:
                    writer.writerows(write_buffer)
                    write_buffer.clear()

                if checkpoint_path:
                    checkpoint_path.write_text(str(raw_lnum))

    except Exception as e:
        stats["error"] = str(e)

    stats["elapsed"] = time.perf_counter() - t0
    return stats
