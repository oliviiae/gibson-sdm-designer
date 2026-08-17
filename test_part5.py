"""
Tests for Part 5 — Flanking primers A and D.

A ~2000 bp test sequence is built deterministically:
  - Left flank  : 895 bp pseudo-random sequence (seed 42)
  - Core region : the same 210 bp sequence used in Parts 1-4
  - Right flank : 895 bp pseudo-random sequence (seed 99)

The mutation region (K15E in the ORF) sits in the middle of the construct,
leaving ≥895 bp on each side for the 700-1000 bp primer-A/D search windows.
"""

import random
from Bio.Seq import Seq

from codon_utils import find_codon_and_mutate
from tm_utils import simple_tm
from primer_bc import design_bc_primers
from primer_ad import (
    find_unique_sites_in_window,
    design_flanking_primer,
    design_ad_primers,
    FlankingPrimerResult,
    CandidateSite,
)

# ---------------------------------------------------------------------------
# Build the 2000 bp test sequence
# ---------------------------------------------------------------------------

_ORF_START_IN_CORE = 10
_ORF_CODONS = [
    "ATG","GCT","GAA","CGT","TTC","CAG","TAT","GGC","ACT","CTG",
    "AGC","GAT","CCG","TGG","AAA","GTC","CAC","TGC","ATC","GAG",
    "GCG","TTT","CGC","AGT","ACC","CAT","GGT","TAC","CTT","GAC",
    "AAG","CCT","GCC","TGT","ATA","GAA","GCT","TTC","CGG","AGC",
    "ACG","CAG","GGC","TAT","CTG","GAT","AAA","CCG","TGG","GTC",
    "CAC","TGC","ATC","GAG","GCG","TTT","CGC","AGT","ACC","CAT",
    "TAA",
]
_CORE_SEQ = "ATCGATCGAT" + "".join(_ORF_CODONS) + "GCTAGCTAGCTAGCTAG"


def _make_flank(length: int, seed: int) -> str:
    rng = random.Random(seed)
    bases = "ACGT"
    return "".join(rng.choice(bases) for _ in range(length))


_LEFT_FLANK  = _make_flank(895, seed=42)
_RIGHT_FLANK = _make_flank(895, seed=99)

LONG_SEQ = _LEFT_FLANK + _CORE_SEQ + _RIGHT_FLANK

# Absolute ORF start in the full 2000 bp sequence
_ORF_START = 895 + _ORF_START_IN_CORE   # 895 + 10 = 905

assert len(LONG_SEQ) == 2000, f"Expected 2000 bp, got {len(LONG_SEQ)}"
assert LONG_SEQ[_ORF_START: _ORF_START + 3] == "ATG", "ORF start codon check"


# ---------------------------------------------------------------------------
# Helper: run Parts 1 and 4 to get b_start / c_end for the long sequence
# ---------------------------------------------------------------------------

def _setup_bc() -> tuple[str, list[int], int, int]:
    """
    Returns (mutated_long_seq, changed_positions, b_start, c_end).
    """
    mutated_seq, changed_pos, _, _ = find_codon_and_mutate(
        LONG_SEQ, _ORF_START, aa_position=15, new_aa="E"
    )
    bc = design_bc_primers(mutated_seq, changed_pos)
    return mutated_seq, changed_pos, bc.b_start, bc.c_end


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_unique_sites_in_window():
    print("\n[Part 5] find_unique_sites_in_window")
    mutated_seq, _, b_start, c_end = _setup_bc()

    # Upstream window (primer A region)
    a_win_start = max(0, b_start - 1000)
    a_win_end   = max(0, b_start -  700)
    a_sites = find_unique_sites_in_window(mutated_seq, a_win_start, a_win_end)

    # Downstream window (primer D region)
    d_win_start = min(len(mutated_seq), c_end + 700)
    d_win_end   = min(len(mutated_seq), c_end + 1000)
    d_sites = find_unique_sites_in_window(mutated_seq, d_win_start, d_win_end)

    print(f"  Upstream window   [{a_win_start}, {a_win_end}): "
          f"{len(a_sites)} unique sites")
    for name, pos in list(a_sites.items())[:5]:
        print(f"    {name}: cut at 0-based {pos[0]}")

    print(f"  Downstream window [{d_win_start}, {d_win_end}): "
          f"{len(d_sites)} unique sites")
    for name, pos in list(d_sites.items())[:5]:
        print(f"    {name}: cut at 0-based {pos[0]}")

    # Every returned site must be unique in the full sequence
    from restriction_utils import scan_sites
    full_map = scan_sites(mutated_seq)
    for name, pos_list in {**a_sites, **d_sites}.items():
        assert len(full_map.get(name, [])) == 1, \
            f"{name} is not unique in full sequence"

    print("  All returned sites verified unique in full construct  PASS")


def test_design_flanking_primer():
    print("\n[Part 5] design_flanking_primer")
    mutated_seq, _, b_start, _ = _setup_bc()

    # Pick an arbitrary position in the upstream window as a mock cut site
    mock_cut = b_start - 850   # well inside the 700-1000 bp window
    mock_cut = max(0, mock_cut)

    primer, start, end, tm, extended = design_flanking_primer(
        mutated_seq, mock_cut, direction=+1
    )
    print(f"  Primer A mock: 5'-{primer}-3'  [{start},{end})  Tm={tm}")
    assert start == mock_cut, "Primer A must start at the cut site"
    assert end > start
    # Per protocol, A/D only require Tm 48-54°C and >=16nt overlap with the
    # vector — no G/C-ending requirement (that's specific to B/C, step 1b).
    assert len(primer) >= 16
    print("  Primer A mock  PASS")

    # Same for a mock cut in the downstream window
    _, _, _, c_end = _setup_bc()
    mock_cut_d = c_end + 850
    mock_cut_d = min(len(mutated_seq) - 1, mock_cut_d)
    primer_d, start_d, end_d, tm_d, extended_d = design_flanking_primer(
        mutated_seq, mock_cut_d, direction=-1
    )
    print(f"  Primer D mock: 5'-{primer_d}-3'  [{start_d},{end_d})  Tm={tm_d}")
    assert end_d - 1 == mock_cut_d or end_d == mock_cut_d + 1, \
        "Primer D must be anchored at the cut site"
    assert len(primer_d) >= 16
    print("  Primer D mock  PASS")


def test_design_ad_primers_full():
    print("\n[Part 5] design_ad_primers — K15E on 2000 bp construct")
    mutated_seq, changed_pos, b_start, c_end = _setup_bc()

    result_a, result_d = design_ad_primers(
        mutated_seq,
        b_start=b_start,
        c_end=c_end,
        tm_range=(48, 54),
        window_bp=(700, 1000),
        min_len=16,
        target_tm=51.0,
    )

    def _print_result(res: FlankingPrimerResult):
        print(f"\n  ── Primer {res.label} candidates "
              f"(window [{res.window_start}, {res.window_end})) ─────")
        if not res.candidates:
            print("    No unique restriction sites found in window.")
            return
        print(f"  Total candidates: {len(res.candidates)}")
        print(f"  {'Rank':<5} {'Enzyme':<14} {'Cut(1-based)':<14} "
              f"{'Primer Tm':>10}  Sequence")
        print(f"  {'-'*5} {'-'*14} {'-'*14} {'-'*10}  {'-'*30}")
        for rank, c in enumerate(res.candidates[:10], 1):
            print(f"  {rank:<5} {c.enzyme:<14} {c.cut_pos:<14} "
                  f"{c.primer_tm:>8.0f}°C  5'-{c.primer_seq}-3'")
        top = res.top
        if top:
            print(f"\n  Top pick: {top.enzyme} at cut pos {top.cut_pos} "
                  f"(1-based), Tm={top.primer_tm}°C")
            print(f"  Primer {res.label}: 5'-{top.primer_seq}-3'  "
                  f"[{top.primer_start}, {top.primer_end})  "
                  f"len={len(top.primer_seq)} bp")

    _print_result(result_a)
    _print_result(result_d)

    # --- Structural assertions -----------------------------------------------
    near, far = 700, 1000
    exp_a_end   = b_start - near
    exp_a_start = b_start - far
    exp_d_start = c_end + near
    exp_d_end   = c_end + far

    assert result_a.window_start == max(0, exp_a_start), "A window start"
    assert result_a.window_end   == max(0, exp_a_end),   "A window end"
    assert result_d.window_start == min(len(mutated_seq), exp_d_start), "D window start"
    assert result_d.window_end   == min(len(mutated_seq), exp_d_end),   "D window end"

    for res in (result_a, result_d):
        if res.candidates:
            top = res.top
            # Primer must be at least 16 bp
            assert len(top.primer_seq) >= 16, \
                f"Primer {res.label} shorter than 16 bp"
            # Primer must start (A) or end (D) at the cut position
            if res.label == "A":
                assert top.primer_start == top.cut_pos_0, \
                    "Primer A must start at cut site"
            else:
                # For direction=-1 the anchor is the rightmost base
                assert top.primer_end - 1 == top.cut_pos_0, \
                    "Primer D must end at cut site"

    print("\n  Structural assertions  PASS")


def test_window_boundary_clamping():
    """Construct too short on one side should clamp windows to sequence edges."""
    print("\n[Part 5] window boundary clamping")
    short_seq = LONG_SEQ[:300]  # only 300 bp — no room for 700 bp upstream window
    mutated_short, changed_pos, _, _ = find_codon_and_mutate(
        # Use a position early in the short sequence by faking a 30 bp ORF start
        short_seq, 10, 1, "E"
    ) if False else (short_seq, [15], 15, 200)   # mock — no real ORF here

    # Only test that design_ad_primers doesn't raise for a tiny b_start
    result_a, result_d = design_ad_primers(
        short_seq, b_start=50, c_end=60,
        window_bp=(700, 1000),
    )
    assert result_a.window_start == 0, "Clamped to 0"
    assert result_d.window_end == len(short_seq), "Clamped to sequence length"
    print(f"  A window: [{result_a.window_start}, {result_a.window_end})  "
          f"D window: [{result_d.window_start}, {result_d.window_end})  PASS")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Full construct: {len(LONG_SEQ)} bp")
    print(f"ORF start     : {_ORF_START} (absolute)")
    print(f"Mutation codon: pos {_ORF_START + 14*3} (K15 = AAA)")

    test_unique_sites_in_window()
    test_design_flanking_primer()
    test_design_ad_primers_full()
    test_window_boundary_clamping()

    print("\nAll Part 5 tests passed.")
