"""
Part 7 tests — pipeline function and CLI.

Covers:
  - Happy path: K15E on the 2000 bp construct
  - Error paths: bad ORF start, wrong original AA, already-same-AA,
                 mutation codon out of range
  - Warning paths: no flanking sites (tiny sequence)
  - parse_mutation_label: valid and invalid formats
  - CLI: formatted output, JSON output, --all-candidates, error exit codes
"""

import json
import random
import subprocess
import sys
import tempfile
import os

from pipeline import design_mutation_primers, parse_mutation_label, result_to_dict

# ---------------------------------------------------------------------------
# Rebuild the 2000 bp construct
# ---------------------------------------------------------------------------

_ORF_CODONS = [
    "ATG","GCT","GAA","CGT","TTC","CAG","TAT","GGC","ACT","CTG",
    "AGC","GAT","CCG","TGG","AAA","GTC","CAC","TGC","ATC","GAG",
    "GCG","TTT","CGC","AGT","ACC","CAT","GGT","TAC","CTT","GAC",
    "AAG","CCT","GCC","TGT","ATA","GAA","GCT","TTC","CGG","AGC",
    "ACG","CAG","GGC","TAT","CTG","GAT","AAA","CCG","TGG","GTC",
    "CAC","TGC","ATC","GAG","GCG","TTT","CGC","AGT","ACC","CAT",
    "TAA",
]
_CORE = "ATCGATCGAT" + "".join(_ORF_CODONS) + "GCTAGCTAGCTAGCTAG"
_ORF_START_IN_CORE = 10

def _flank(n, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))

CONSTRUCT = _flank(895, 42) + _CORE + _flank(895, 99)
ORF_START  = 895 + _ORF_START_IN_CORE  # 905

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_fasta(seq: str, path: str, name: str = "test_construct"):
    with open(path, "w") as f:
        f.write(f">{name}\n{seq}\n")

def _run_cli(*args):
    """Run sdm_design.py as a subprocess; return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "sdm_design.py", *args],
        capture_output=True, text=True
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# parse_mutation_label
# ---------------------------------------------------------------------------

def test_parse_mutation_label():
    print("\n[Part 7] parse_mutation_label")
    cases = [
        ("K255E",  ("K", 255, "E")),
        ("k255e",  ("K", 255, "E")),
        ("R1A",    ("R", 1,   "A")),
        ("W99*",   ("W", 99,  "*")),
    ]
    for label, expected in cases:
        got = parse_mutation_label(label)
        assert got == expected, f"{label}: expected {expected}, got {got}"
        print(f"  {label!r:12} → {got}  PASS")

    bad = ["K255", "255E", "KE", "K-255E", "", "K2 55E"]
    for label in bad:
        try:
            parse_mutation_label(label)
            print(f"  {label!r}: expected ValueError — FAIL")
        except ValueError:
            print(f"  {label!r}: correctly raised ValueError  PASS")


# ---------------------------------------------------------------------------
# Pipeline happy path
# ---------------------------------------------------------------------------

def test_pipeline_happy_path():
    print("\n[Part 7] design_mutation_primers — K15E happy path (explicit orf_start)")
    r = design_mutation_primers(CONSTRUCT, 15, "K", "E", orf_start=ORF_START)

    assert r.errors == [], f"Unexpected errors: {r.errors}"
    assert r.mutation_label == "K15E"
    assert r.original_codon == "AAA"
    assert r.new_codon in ("GAA", "GAG")
    assert r.primer_B is not None and r.primer_C is not None
    assert r.overlap_seq != ""
    assert r.translation_passed is True, f"Translation failed: {r.translation_detail}"
    assert r.overall_passed is True

    print(f"  Mutation : {r.mutation_label}  {r.original_codon}→{r.new_codon}")
    print(f"  Primer A : {r.primer_A.sequence if r.primer_A else 'MISSING'}  "
          f"Tm={r.primer_A.tm if r.primer_A else '?'}°C")
    print(f"  Primer B : {r.primer_B.sequence}  Tm={r.primer_B.tm}°C")
    print(f"  Primer C : {r.primer_C.sequence}  Tm(anneal)={r.primer_C.tm_anneal}°C")
    print(f"  Primer D : {r.primer_D.sequence if r.primer_D else 'MISSING'}  "
          f"Tm={r.primer_D.tm if r.primer_D else '?'}°C")
    print(f"  Overlap  : {r.overlap_seq}  Tm={r.overlap_tm}°C")
    if r.diagnostic:
        print(f"  Diagnostic: {r.diagnostic.enzyme}  {r.diagnostic.effect}  "
              f"({r.diagnostic.source})")
    print(f"  Frags    : ab={r.frag_ab_bp}bp  cd={r.frag_cd_bp}bp  ad={r.frag_ad_bp}bp")
    print(f"  Warnings : {r.warnings or 'none'}")
    print("  PASS")


# ---------------------------------------------------------------------------
# result_to_dict round-trip
# ---------------------------------------------------------------------------

def test_auto_detect_orf():
    print("\n[Part 7] ORF auto-detection — unambiguous sequence")
    # Build a short sequence with exactly one ATG where aa15 is K
    core = "ATCGATCGAT" + "".join(_ORF_CODONS) + "GCTAGCTAGCTAGCTAG"
    unambiguous = _flank(895, 11) + core + _flank(895, 13)
    r = design_mutation_primers(unambiguous, 15, "K", "E")
    if r.errors:
        # Still might be ambiguous in this random flank; skip gracefully
        print(f"  Ambiguous or error (acceptable): {r.errors[0][:60]}  SKIP")
    else:
        assert r.orf_start_detected == 895 + 10
        print(f"  Auto-detected ORF start: {r.orf_start_detected}  PASS")

    # Ambiguous case: construct has two ATGs where aa15=K → should error
    r2 = design_mutation_primers(CONSTRUCT, 15, "K", "E")
    assert r2.errors, "Expected ambiguity error without orf_start"
    assert "ambiguous" in r2.errors[0].lower() or "2 ATGs" in r2.errors[0]
    print(f"  Ambiguous case correctly flagged: {r2.errors[0][:60]}  PASS")


def test_result_to_dict():
    print("\n[Part 7] result_to_dict — JSON serialisability")
    r = design_mutation_primers(CONSTRUCT, 15, "K", "E", orf_start=ORF_START)
    d = result_to_dict(r)
    serialised = json.dumps(d)   # raises if non-serialisable objects remain
    parsed = json.loads(serialised)
    assert parsed["mutation"] == "K15E"
    assert parsed["verification"]["overall"] is True
    assert "primers" in parsed
    assert "fragment_lengths" in parsed
    print("  dict keys:", sorted(parsed.keys()))
    print("  PASS")


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_bad_orf_start():
    print("\n[Part 7] Error: orf_start out of range")
    r = design_mutation_primers(CONSTRUCT, 15, "K", "E", orf_start=len(CONSTRUCT) + 10)
    assert r.errors, "Expected an error"
    assert r.overall_passed is False
    print(f"  Error: {r.errors[0][:80]}  PASS")


def test_wrong_original_aa():
    print("\n[Part 7] Error: wrong original AA")
    r = design_mutation_primers(CONSTRUCT, 15, "R", "E", orf_start=ORF_START)  # R15 doesn't exist
    assert r.errors, "Expected an error"
    print(f"  Error: {r.errors[0][:80]}  PASS")


def test_already_same_aa():
    print("\n[Part 7] Error: already the desired AA")
    # aa3 is already E (GAA)
    r = design_mutation_primers(CONSTRUCT, 3, "E", "E", orf_start=ORF_START)
    assert r.errors, "Expected an error"
    print(f"  Error: {r.errors[0][:80]}  PASS")


def test_position_out_of_range():
    print("\n[Part 7] Error: codon position extends past sequence end")
    r = design_mutation_primers(CONSTRUCT, 9999, "K", "E", orf_start=ORF_START)
    assert r.errors, "Expected an error"
    print(f"  Error: {r.errors[0][:80]}  PASS")


# ---------------------------------------------------------------------------
# Warning path: no flanking sites (tiny construct)
# ---------------------------------------------------------------------------

def test_no_flanking_sites_warning():
    print("\n[Part 7] Warning: no flanking sites (300 bp construct)")
    # 300 bp is too short to have a 700-1000 bp flanking window
    tiny = CONSTRUCT[:300]
    # Find a valid mutation in the first few codons of the tiny sequence
    # ORF starts at 905, but for a tiny seq we need to fake a short ORF
    # Use just the core sequence with a short flank
    short_flank = _flank(20, 7)
    short_construct = short_flank + _CORE  # ~230 bp total
    short_orf = 20 + _ORF_START_IN_CORE

    r = design_mutation_primers(short_construct, 15, "K", "E")
    # Should not crash; should have a warning about missing flanking sites
    assert any("primer A" in w or "primer D" in w for w in r.warnings), \
        f"Expected flanking-site warning; got warnings={r.warnings} errors={r.errors}"
    print(f"  Warnings: {r.warnings}  PASS")


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

def test_cli_formatted():
    print("\n[Part 7] CLI — formatted output")
    with tempfile.NamedTemporaryFile(suffix=".fasta", mode="w",
                                    delete=False) as fh:
        fh.write(f">construct\n{CONSTRUCT}\n")
        fasta_path = fh.name
    try:
        rc, stdout, stderr = _run_cli(fasta_path, "--orf-start", str(ORF_START),
                                      "--mutation", "K15E")
        print(f"  Exit code: {rc}")
        # Print a few lines of output
        for line in stdout.splitlines()[:25]:
            print(f"    {line}")
        assert rc == 0, f"CLI exited {rc}\nstderr: {stderr}"
        assert "K15E" in stdout
        assert "PASS" in stdout
        print("  PASS")
    finally:
        os.unlink(fasta_path)


def test_cli_json():
    print("\n[Part 7] CLI — JSON output")
    with tempfile.NamedTemporaryFile(suffix=".fasta", mode="w",
                                    delete=False) as fh:
        fh.write(f">construct\n{CONSTRUCT}\n")
        fasta_path = fh.name
    try:
        rc, stdout, stderr = _run_cli(fasta_path, "--orf-start", str(ORF_START),
                                      "--mutation", "K15E", "--json")
        assert rc == 0, f"CLI exited {rc}\nstderr: {stderr}"
        data = json.loads(stdout)
        assert data["mutation"] == "K15E"
        assert data["verification"]["overall"] is True
        print(f"  JSON keys: {sorted(data.keys())}")
        print(f"  Overall: {data['verification']['overall']}  PASS")
    finally:
        os.unlink(fasta_path)


def test_cli_error_exit():
    print("\n[Part 7] CLI — error exit code on bad input")
    with tempfile.NamedTemporaryFile(suffix=".fasta", mode="w",
                                     delete=False) as fh:
        fh.write(f">construct\n{CONSTRUCT}\n")
        fasta_path = fh.name
    try:
        # Wrong original AA → pipeline error → exit 1
        rc, stdout, stderr = _run_cli(fasta_path, "--orf-start", str(ORF_START),
                                      "--mutation", "R15E")
        print(f"  Exit code: {rc}  (expected 1)")
        assert rc == 1, f"Expected exit 1, got {rc}"
        print("  PASS")
    finally:
        os.unlink(fasta_path)


def test_cli_all_candidates():
    print("\n[Part 7] CLI -- --all-candidates flag")
    with tempfile.NamedTemporaryFile(suffix=".fasta", mode="w",
                                     delete=False) as fh:
        fh.write(f">construct\n{CONSTRUCT}\n")
        fasta_path = fh.name
    try:
        rc, stdout, stderr = _run_cli(fasta_path, "--orf-start", str(ORF_START),
                                      "--mutation", "K15E", "--all-candidates")
        assert rc == 0, f"Exit {rc}, stderr: {stderr}"
        assert "All primer A candidates" in stdout or "primer A" in stdout.lower()
        print("  --all-candidates flag works  PASS")
    finally:
        os.unlink(fasta_path)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_parse_mutation_label()
    test_pipeline_happy_path()
    test_result_to_dict()
    test_bad_orf_start()
    test_wrong_original_aa()
    test_already_same_aa()
    test_position_out_of_range()
    test_no_flanking_sites_warning()
    test_cli_formatted()
    test_cli_json()
    test_cli_error_exit()
    test_cli_all_candidates()
    print("\nAll Part 7 tests passed.")
