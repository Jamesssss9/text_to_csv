"""
Worker function executed inside multiprocessing.Pool.
Each worker handles one file at a time.
"""

import os
from pathlib import Path

from src.converter import convert_file


def worker_convert(args: tuple) -> dict:
    src_str, dst_str, checkpoint_str, cfg = args
    src_path = Path(src_str)
    dst_path = Path(dst_str)
    checkpoint_path = Path(checkpoint_str) if checkpoint_str else None

    result = convert_file(src_path, dst_path, cfg, checkpoint_path)
    result["src"] = src_str
    result["dst"] = dst_str
    return result
