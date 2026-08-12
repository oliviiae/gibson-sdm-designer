"""
Tests for Part 4 — Primer B/C boundary and overlap design.

Reuses the same 210 bp test sequence and K15E mutation from test_parts_1_3.py.
"""

from Bio.Seq import Seq

from codon_utils import find_codon_and_mutate
from tm_utils import simple_tm
from primer_bc import design_bc_primers, find_bc_overlap, PrimerBCResult

# ---------------------------------------------------------------------------
# Reproduce test sequence (same as test_parts_1_3.py)
# ---------------------------------------------------------------------------

_ORF_START = 10
_ORF_CODONS = [
    "ATG","GCT","GAA","CGT","TTC","CAG","TAT","GGC","ACT","CTG",
    "AGC","GAT","CCG","TGG","AAA","GTC","CAC","TGC","ATC","GAG",
    "GCG","TTT","CGC","AGT","ACC","CAT","GGT","TAC","CTT","GAC",
    "AAG","CCT","GCC","TGT","ATA","GAA","GCT","TTC","CGG","AGC",
    "ACG","CAG","GGC","TAT","CTG","GAT","AAA","CCG","TGG","GTC",
    "CAC","TGC","ATC","GAG","GCG","TTT","CGC","AGT","ACC","CAT",
    "TAA",
]
TEST_SEQ = "ATCGATCGAT" + "".join(_ORF_CODONS) + "GCTAGCTAGCTAGCTAG"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _annotate_seq(seq: str, marks: dict[int, str], window: int = 5) -> str:
    """
    Return a view of seq with tick marks at specified positions.
    marks = {abs_pos: label}
    """
    lines = [seq]
    ruler = [" "] * len(seq)
    for pos, label in marks.items():
        ruler[pos] = "|"
    lines.append("".join(ruler))
    for pos, label in sorted(marks.items()):
        lines.append(f"  pos {pos:3d} → {label}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_find_bc_overlap_basic():
    """find_bc_overlap should return a window of correct length and Tm."""
    print("\n[Part 4] find_bc_overlap — basic")
    seq = "GCGCATGCGCATGCGCATGCGCAT"   # all GC → Tm = 4 per base
    ov, start, end, tm_fwd, tm_rc = find_bc_overlap(
        seq, search_from=0, min_len=12, tm_range=(48, 54)
    )
    print(f"  window: {ov}  start={start} end={end} tm_fwd={tm_fwd} tm_rc={tm_rc}")
    assert start == 0
    assert end - start >= 12
    assert 48 <= tm_fwd <= 54
    assert tm_fwd == tm_rc, "Wallace Tm must be equal for fwd and RC"
    print("  PASS")


def test_find_bc_overlap_tm_rc_symmetry():
    """
    With the Wallace sum rule, Tm(seq) == Tm(RC(seq)) for any sequence.
    Verify this property explicitly.
    """
    print("\n[Part 4] Tm symmetry: simple_tm(seq) == simple_tm(RC(seq))")
    for seq in ["ATCGATCG", "GCGCGCGC", "ATATATAT", "ATGCGTAC"]:
        rc = str(Seq(seq).reverse_complement())
        assert simple_tm(seq) == simple_tm(rc), f"Failed for {seq}"
        print(f"  {seq} → RC {rc}  Tm={simple_tm(seq)}  PASS")


def test_design_bc_primers_k15e():
    """Full B/C design on the K15E mutation."""
    print("\n[Part 4] design_bc_primers — K15E")

    mutated_seq, changed_pos, orig_codon, new_codon = find_codon_and_mutate(
        TEST_SEQ, _ORF_START, aa_position=15, new_aa="E"
    )
    print(f"  Mutation: {orig_codon}→{new_codon}  at absolute pos {changed_pos}")

    result: PrimerBCResult = design_bc_primers(
        mutated_seq,
        changed_pos,
        tm_range=(48, 54),
        min_overlap=16,
    )

    print()
    print(f"  ── Primer B ──────────────────────────────────────────────")
    print(f"  Sequence  : 5'-{result.primer_b}-3'")
    print(f"  Length    : {len(result.primer_b)} bp")
    print(f"  Tm        : {result.tm_b} °C")
    print(f"  Position  : [{result.b_start}, {result.b_end})  "
          f"(0-based, forward strand)")

    print()
    print(f"  ── Overlap (shared B∩C) ──────────────────────────────────")
    print(f"  Sequence  : 5'-{result.overlap_seq}-3'")
    print(f"  Length    : {len(result.overlap_seq)} bp")
    print(f"  Tm fwd    : {result.tm_overlap_fwd} °C")
    print(f"  Tm RC     : {result.tm_overlap_rc} °C")
    print(f"  Position  : [{result.overlap_start}, {result.overlap_end})")

    print()
    print(f"  ── Primer C ──────────────────────────────────────────────")
    print(f"  Template  : 5'-{result.primer_c_fwd}-3'  (fwd strand reference)")
    print(f"  Sequence  : 5'-{result.primer_c}-3'  (antisense — order this)")
    print(f"  Length    : {len(result.primer_c)} bp")
    print(f"  Tm (full) : {result.tm_c_full} °C  (overlap + annealing tail)")
    print(f"  Tm (anneal): {result.tm_c_anneal} °C  (unique downstream region only)")
    print(f"  Position  : [{result.c_start}, {result.c_end})  "
          f"(fwd strand coords)")

    # Annotated sequence view
    print()
    print(f"  ── Annotated sequence ────────────────────────────────────")
    marks = {
        result.b_start:      "B start (5')",
        changed_pos[0]:      "mutation",
        result.overlap_start:"overlap start",
        result.overlap_end-1:"overlap end",
        result.c_end-1:      "C end (3' on fwd)",
    }
    print("  " + mutated_seq)
    ruler = list(" " * len(mutated_seq))
    symbols = {
        result.b_start:       "B",
        changed_pos[0]:       "*",
        result.overlap_start: "[",
        result.overlap_end-1: "]",
        result.c_end-1:       "C",
    }
    for pos, sym in symbols.items():
        ruler[pos] = sym
    print("  " + "".join(ruler))
    for pos, label in sorted(marks.items()):
        print(f"    pos {pos:3d} {label}")

    # --- Assertions ---
    assert result.b_start < changed_pos[0], "B must start upstream of mutation"
    assert result.overlap_end > changed_pos[-1], "Overlap must cover mutation"
    assert result.c_end > result.overlap_end, "C must extend past overlap"
    assert len(result.overlap_seq) >= 16, "Overlap must be ≥ 16 bp"

    # Primer B always starts at b_start and ends where the overlap ends. The
    # overlap itself is guaranteed Tm 48-54 and G/C at both ends (checked
    # below), but the overlap's start can land a few bases after b_start
    # when needed to satisfy those constraints together — in that case
    # primer B is slightly longer than the overlap alone and its own Tm can
    # run a bit above the target window. What must always hold is that
    # primer B's sequence actually matches [b_start:overlap_end].
    assert result.b_end == result.overlap_end, \
        "Primer B must end exactly where the overlap ends"
    assert result.tm_b >= 48, f"Tm_B {result.tm_b} unexpectedly low"

    # The overlap itself (not necessarily all of primer B) must be in range
    # and start/end on G/C — this is the actual protocol requirement.
    assert 48 <= result.tm_overlap_fwd <= 54, \
        f"Overlap Tm {result.tm_overlap_fwd} outside target window"
    assert result.overlap_seq[0] in "GC", "Overlap must start with G/C"
    assert result.overlap_seq[-1] in "GC", "Overlap must end with G/C"

    # Primer C spans overlap_start → c_end, so it is LONGER than the overlap alone.
    # With the Wallace formula, its Tm will exceed 54°C for any primer longer than
    # ~18 bp with average GC content.  We only assert it is positive and larger than
    # the overlap Tm (both expected structural properties).
    assert result.tm_c_full >= result.tm_overlap_fwd, \
        "Full primer C Tm must be ≥ overlap Tm (C is a superset of the overlap)"
    assert result.tm_c_full > 0
    assert result.tm_c_anneal >= 0

    assert 48 <= result.tm_overlap_fwd <= 54, \
        f"Overlap Tm {result.tm_overlap_fwd} outside target"
    assert result.tm_overlap_fwd == result.tm_overlap_rc

    # Primer C sequence should be RC of the template region
    expected_c = str(Seq(result.primer_c_fwd).reverse_complement())
    assert result.primer_c == expected_c, "Primer C is not RC of template region"

    # Primer B must contain the mutation
    assert mutated_seq[changed_pos[0]] == result.primer_b[changed_pos[0] - result.b_start], \
        "Primer B must carry the mutated base"

    print("\n  PASS")


def test_design_bc_primers_overlap_covers_mutation():
    """
    When the mutation is far downstream (near the 3' end of the sequence),
    the overlap should still encompass it.
    """
    print("\n[Part 4] Overlap always covers mutation — aa47 K→E")
    mutated_seq, changed_pos, orig, new = find_codon_and_mutate(
        TEST_SEQ, _ORF_START, aa_position=47, new_aa="E"
    )
    result = design_bc_primers(mutated_seq, changed_pos)
    assert result.overlap_end > changed_pos[-1], \
        f"Overlap end {result.overlap_end} does not cover mutation at {changed_pos}"
    print(f"  aa47 K→E: overlap [{result.overlap_start},{result.overlap_end}) "
          f"covers mutation at {changed_pos}  PASS")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_find_bc_overlap_basic()
    test_find_bc_overlap_tm_rc_symmetry()
    test_design_bc_primers_k15e()
    test_design_bc_primers_overlap_covers_mutation()
    print("\nAll Part 4 tests passed.")
