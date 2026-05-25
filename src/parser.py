"""
Detects and parses various text formats into row lists.
Modes: auto, delimited, fixed_width, csv
"""

import re
import csv
import io
from typing import Iterator, List, Optional


def detect_delimiter(sample_lines: List[str]) -> Optional[str]:
    """Sniff delimiter from sample lines using csv.Sniffer, fallback heuristics."""
    sample = "\n".join(sample_lines[:20])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;: ")
        return dialect.delimiter
    except csv.Error:
        pass
    # Heuristic: count candidates
    candidates = ["\t", ",", "|", ";", " "]
    counts = {d: sample.count(d) for d in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else None


def detect_fixed_widths(sample_lines: List[str]) -> Optional[List[int]]:
    """Detect column break positions from whitespace patterns."""
    if not sample_lines:
        return None
    # Find positions where every line has a space (potential column boundary)
    min_len = min(len(l) for l in sample_lines)
    if min_len == 0:
        return None
    always_space = [
        all(i < len(l) and l[i] == " " for l in sample_lines)
        for i in range(min_len)
    ]
    widths = []
    start = 0
    in_space = False
    for i, is_sp in enumerate(always_space):
        if is_sp and not in_space:
            if i > start:
                widths.append(i - start)
            start = i
            in_space = True
        elif not is_sp and in_space:
            in_space = False
    if start < min_len:
        widths.append(min_len - start)
    return widths if len(widths) > 1 else None


class TextParser:
    def __init__(self, cfg: dict):
        self.mode = cfg.get("mode", "auto")
        self.delimiter = cfg.get("delimiter")
        self.fixed_widths = cfg.get("fixed_widths")
        self.skip_blank = cfg.get("skip_blank_lines", True)
        self.skip_comment = cfg.get("skip_comment_lines", True)
        self.comment_prefix = cfg.get("comment_prefix", "#")
        self.header_line = cfg.get("header_line")  # int line index or None
        self.encoding = cfg.get("encoding", "utf-8")
        self.encoding_errors = cfg.get("encoding_errors", "replace")

        self._delimiter = self.delimiter
        self._widths = self.fixed_widths
        self._header: Optional[List[str]] = None
        self._detected = False

    def _filter_line(self, line: str) -> Optional[str]:
        stripped = line.rstrip("\n\r")
        if self.skip_blank and not stripped.strip():
            return None
        if self.skip_comment and stripped.lstrip().startswith(self.comment_prefix):
            return None
        return stripped

    def _split_delimited(self, line: str) -> List[str]:
        return next(csv.reader([line], delimiter=self._delimiter or ","))

    def _split_fixed(self, line: str) -> List[str]:
        cols = []
        pos = 0
        for w in self._widths:
            cols.append(line[pos : pos + w].strip())
            pos += w
        cols.append(line[pos:].strip())
        return cols

    def _auto_detect(self, sample_lines: List[str]):
        fw = detect_fixed_widths(sample_lines)
        if fw:
            self._widths = fw
            self._effective_mode = "fixed_width"
        else:
            d = detect_delimiter(sample_lines)
            self._delimiter = d or ","
            self._effective_mode = "delimited"
        self._detected = True

    def parse_lines(self, lines: List[str]) -> Iterator[List[str]]:
        """Parse a batch of raw lines into rows."""
        filtered = [l for l in (self._filter_line(ln) for ln in lines) if l is not None]

        if not self._detected and self.mode == "auto" and filtered:
            self._auto_detect(filtered[:50])
            self._detected = True

        for line in filtered:
            if self._effective_mode == "fixed_width":
                yield self._split_fixed(line)
            else:
                yield self._split_delimited(line)

    @property
    def _effective_mode(self):
        return getattr(self, "_eff_mode", self.mode)

    @_effective_mode.setter
    def _effective_mode(self, v):
        self._eff_mode = v

    def reset(self):
        """Reset per-file state for reuse across files."""
        if self.mode == "auto":
            self._detected = False
            self._delimiter = self.delimiter
            self._widths = self.fixed_widths
            self._eff_mode = self.mode
