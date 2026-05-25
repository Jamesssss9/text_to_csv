"""
Quick smoke test — generates a sample text file then converts it.
Run: python test_sample.py
"""

import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

# Create a sample input file
sample = BASE / "input" / "sample.txt"
sample.parent.mkdir(exist_ok=True)
with open(sample, "w") as f:
    f.write("# comment line\n")
    f.write("\n")
    f.write("name age city score\n")
    for i in range(1, 101):
        f.write(f"user_{i:04d} {20+i%40} CityX_{i%10} {i*1.5:.2f}\n")

print(f"Created {sample} ({sample.stat().st_size} bytes, 102 lines)")

# Load config
with open(BASE / "config" / "config.json") as f:
    cfg = json.load(f)

# Run converter
from src.converter import convert_file

out = BASE / "output" / "sample.csv"
stats = convert_file(sample, out, cfg)

print(f"Output: {out}")
print(f"Rows:   {stats['rows']}")
print(f"Error:  {stats['error']}")

# Show first 5 lines of result
with open(out) as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(line, end="")

print("\nSmoke test passed!" if not stats["error"] else "FAILED")
