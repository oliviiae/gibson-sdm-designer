"""
.xdna file reader (Serial Cloner / DNA Strider legacy binary format).

This format has no official public specification (unlike SnapGene's .dna
format). Rather than guess at exact byte offsets in the binary header —
which risks silently corrupting the sequence — this reader scans the raw
bytes for the longest contiguous run of valid DNA characters (the sequence
itself is stored as plain ASCII within the binary container in every known
variant of this format family).

IMPORTANT: Always sanity-check the reported sequence length and preview
against your sequence viewer (SnapGene, Serial Cloner, ApE) before using
the result to design primers you intend to order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_DNA_RUN_RE = re.compile(rb"[ACGTNacgtn]{30,}")


@dataclass
class XdnaReadResult:
    sequence: str            # uppercase, cleaned (N's preserved)
    length: int
    file_size: int
    run_start_byte: int       # byte offset in the file where the sequence run starts
    run_end_byte: int
    preview_start: str        # first 60 bases
    preview_end: str          # last 60 bases
    n_count: int               # count of ambiguous 'N' bases
    candidate_runs_found: int  # how many DNA-like runs were in the file total


def read_xdna_sequence(path: str, min_run_len: int = 30) -> XdnaReadResult:
    """
    Extract the sequence from a .xdna file by finding the longest contiguous
    run of DNA characters in the raw bytes.

    Raises ValueError if no run of at least min_run_len bases is found, or if
    multiple runs of similar (within 10%) length are found (ambiguous — could
    mean the file contains multiple sequences, e.g. a feature/primer table
    with embedded short sequences, and we can't be sure which is the construct).
    """
    with open(path, "rb") as f:
        data = f.read()

    runs = [
        (m.start(), m.end(), m.group())
        for m in _DNA_RUN_RE.finditer(data)
        if len(m.group()) >= min_run_len
    ]

    if not runs:
        raise ValueError(
            f"No DNA-like sequence run of at least {min_run_len} bases found "
            f"in {path}. This may not be a valid .xdna file, or the format "
            f"variant differs from what this reader expects."
        )

    runs.sort(key=lambda r: len(r[2]), reverse=True)
    best_start, best_end, best_seq = runs[0]

    # Ambiguity check: if a second run is within 10% of the longest run's
    # length, we can't be confident which one is the actual construct.
    if len(runs) > 1:
        second_len = len(runs[1][2])
        if second_len >= 0.9 * len(best_seq):
            raise ValueError(
                f"Found multiple DNA-like runs of similar length in {path} "
                f"({len(best_seq)} bp and {second_len} bp) — cannot "
                f"confidently determine which is the construct sequence. "
                f"Export to FASTA from your sequence viewer instead."
            )

    seq = best_seq.decode("ascii").upper()

    return XdnaReadResult(
        sequence=seq,
        length=len(seq),
        file_size=len(data),
        run_start_byte=best_start,
        run_end_byte=best_end,
        preview_start=seq[:60],
        preview_end=seq[-60:] if len(seq) > 60 else seq,
        n_count=seq.count("N"),
        candidate_runs_found=len(runs),
    )


def print_verification(result: XdnaReadResult, path: str) -> None:
    """Print a human-checkable summary so the user can confirm before trusting the result."""
    print(f"\n  ⚠  Reading .xdna file: {path}")
    print(f"  This format has no public spec — please verify the numbers below")
    print(f"  against your sequence viewer (SnapGene / Serial Cloner / ApE) before")
    print(f"  ordering any primers based on this sequence.\n")
    print(f"    File size        : {result.file_size} bytes")
    print(f"    Extracted length : {result.length} bp")
    if result.n_count:
        print(f"    Ambiguous (N)    : {result.n_count} base(s)")
    if result.candidate_runs_found > 1:
        print(f"    Note: {result.candidate_runs_found} DNA-like runs found in "
              f"the file; used the longest one.")
    print(f"    First 60 bp      : {result.preview_start}")
    print(f"    Last 60 bp       : {result.preview_end}")
    print()
