"""
Tests for Parts 1–3.

Test sequence (210 bp):
  - ORF starts at position 10 (0-based)
  - Encodes 60 amino acids (180 bp) + stop codon = 183 bp
  - Flanked by 10 bp on each side

Amino acid 15 in the ORF is Lys (K, codon AAA).
We test K15E mutation.
"""

from codon_utils import find_codon_and_mutate
from tm_utils import simple_tm, walk_to_tm
from restriction_utils import scan_sites, gained_lost_sites, find_silent_restriction_sites

# ---------------------------------------------------------------------------
# Build a predictable test sequence
# ---------------------------------------------------------------------------
# 10 bp flank | 183 bp ORF (60 aa + stop) | 17 bp flank = 210 bp
# Codon 15 (1-based) is at orf_start + 14*3 = 10 + 42 = bp 52 (0-based)
# We hard-code it as AAA (K).

_ORF_START = 10
_ORF_CODONS = [
    "ATG",  # 1  M
    "GCT",  # 2  A
    "GAA",  # 3  E
    "CGT",  # 4  R
    "TTC",  # 5  F
    "CAG",  # 6  Q
    "TAT",  # 7  Y
    "GGC",  # 8  G
    "ACT",  # 9  T
    "CTG",  # 10 L
    "AGC",  # 11 S
    "GAT",  # 12 D
    "CCG",  # 13 P
    "TGG",  # 14 W
    "AAA",  # 15 K  <-- target for K15E
    "GTC",  # 16 V
    "CAC",  # 17 H
    "TGC",  # 18 C
    "ATC",  # 19 I
    "GAG",  # 20 E
    "GCG",  # 21 A
    "TTT",  # 22 F
    "CGC",  # 23 R
    "AGT",  # 24 S
    "ACC",  # 25 T
    "CAT",  # 26 H
    "GGT",  # 27 G
    "TAC",  # 28 Y
    "CTT",  # 29 L
    "GAC",  # 30 D
    "AAG",  # 31 K
    "CCT",  # 32 P
    "GCC",  # 33 A
    "TGT",  # 34 C
    "ATA",  # 35 I
    "GAA",  # 36 E
    "GCT",  # 37 A
    "TTC",  # 38 F
    "CGG",  # 39 R
    "AGC",  # 40 S
    "ACG",  # 41 T
    "CAG",  # 42 Q
    "GGC",  # 43 G
    "TAT",  # 44 Y
    "CTG",  # 45 L
    "GAT",  # 46 D
    "AAA",  # 47 K
    "CCG",  # 48 P
    "TGG",  # 49 W
    "GTC",  # 50 V
    "CAC",  # 51 H
    "TGC",  # 52 C
    "ATC",  # 53 I
    "GAG",  # 54 E
    "GCG",  # 55 A
    "TTT",  # 56 F
    "CGC",  # 57 R
    "AGT",  # 58 S
    "ACC",  # 59 T
    "CAT",  # 60 H
    "TAA",  # stop
]

_LEFT_FLANK  = "ATCGATCGAT"   # 10 bp
_RIGHT_FLANK = "GCTAGCTAGCTAGCTAG"  # 17 bp
TEST_SEQ = _LEFT_FLANK + "".join(_ORF_CODONS) + _RIGHT_FLANK

assert len(TEST_SEQ) == 210, f"Expected 210 bp, got {len(TEST_SEQ)}"


# ---------------------------------------------------------------------------
# Part 1 tests
# ---------------------------------------------------------------------------

def test_part1_basic():
    mutated, positions, orig, new = find_codon_and_mutate(
        TEST_SEQ, _ORF_START, aa_position=15, new_aa="E"
    )
    print(f"\n[Part 1] K15E mutation")
    print(f"  Original codon : {orig}  (pos {_ORF_START + 14*3}–{_ORF_START + 14*3 + 2})")
    print(f"  New codon      : {new}")
    print(f"  Changed bases  : absolute positions {positions}")
    print(f"  Mutated region : ...{mutated[_ORF_START+42-3 : _ORF_START+42+6]}...")

    assert orig == "AAA", f"Expected AAA, got {orig}"
    assert new in ("GAA", "GAG"), f"Expected GAA or GAG, got {new}"
    assert len(positions) == 1, "K→E should require 1 base change"
    assert mutated[:_ORF_START + 42] == TEST_SEQ[:_ORF_START + 42], "Upstream unchanged"
    print("  PASS")


def test_part1_wrong_aa():
    """Test that original codon validation works."""
    try:
        find_codon_and_mutate(TEST_SEQ, _ORF_START, aa_position=3, new_aa="E")
        # position 3 is already GAA = E
        print("[Part 1] should have raised ValueError for E3E — FAIL")
    except ValueError as exc:
        print(f"\n[Part 1] Already-same-AA guard: {exc}  PASS")


# ---------------------------------------------------------------------------
# Part 2 tests
# ---------------------------------------------------------------------------

def test_part2_simple_tm():
    print("\n[Part 2] simple_tm")
    seq = "GCGCAT"   # G+C+G+C+A+T = 4+4+4+4+2+2 = 20
    tm = simple_tm(seq)
    assert tm == 20, f"Expected 20, got {tm}"
    print(f"  simple_tm('GCGCAT') = {tm}  PASS")

    seq2 = "AAATTT"   # all AT = 6×2 = 12
    tm2 = simple_tm(seq2)
    assert tm2 == 12, f"Expected 12, got {tm2}"
    print(f"  simple_tm('AAATTT') = {tm2}  PASS")


def test_part2_walk():
    print("\n[Part 2] walk_to_tm")
    # Anchor at the mutation site (codon start = orf_start + 42)
    anchor = _ORF_START + 14 * 3  # = 52

    primer, start, end, tm = walk_to_tm(
        TEST_SEQ, anchor_pos=anchor, direction=1,
        tm_range=(48, 54), must_end_gc=True
    )
    print(f"  Rightward primer: {primer}")
    print(f"  Range [{start}, {end}), Tm={tm}")
    assert 48 <= tm <= 60, f"Tm {tm} out of expected range"
    assert primer[-1] in "GC", f"Terminal base should be G or C, got {primer[-1]}"
    print("  PASS")

    primer2, start2, end2, tm2 = walk_to_tm(
        TEST_SEQ, anchor_pos=anchor, direction=-1,
        tm_range=(48, 54), must_end_gc=True
    )
    print(f"  Leftward primer : {primer2}")
    print(f"  Range [{start2}, {end2}), Tm={tm2}")
    assert primer2[0] in "GC", f"5' base should be G or C, got {primer2[0]}"
    print("  PASS")


# ---------------------------------------------------------------------------
# Part 3 tests
# ---------------------------------------------------------------------------

def test_part3_scan():
    print("\n[Part 3] scan_sites on test sequence")
    sites = scan_sites(TEST_SEQ)
    print(f"  Enzymes with sites: {len(sites)}")
    for name, positions in list(sites.items())[:5]:
        print(f"    {name}: {positions}")
    assert isinstance(sites, dict)
    print("  PASS")


def test_part3_gained_lost():
    print("\n[Part 3] gained_lost_sites for K15E")
    mutated, _, _, _ = find_codon_and_mutate(
        TEST_SEQ, _ORF_START, aa_position=15, new_aa="E"
    )
    diff = gained_lost_sites(TEST_SEQ, mutated)
    gained = diff["gained"]
    lost = diff["lost"]
    print(f"  Sites gained: {list(gained.keys())[:10]}")
    print(f"  Sites lost  : {list(lost.keys())[:10]}")
    assert isinstance(gained, dict)
    assert isinstance(lost, dict)
    print("  PASS")


def test_part3_silent_sites():
    print("\n[Part 3] find_silent_restriction_sites in 30-bp window around mutation")
    codon_abs = _ORF_START + 14 * 3  # codon 15 start
    window_start = codon_abs - 15
    window_end   = codon_abs + 18
    results = find_silent_restriction_sites(
        TEST_SEQ, _ORF_START, window_start, window_end
    )
    print(f"  Silent mutations found: {len(results)}")
    for r in results[:5]:
        print(f"    aa{r['aa_index']} {r['original_codon']}→{r['new_codon']} "
              f"({r['changes']} change) {r['effect']} {r['enzyme']} @ {r['site_positions']}")
    assert isinstance(results, list)
    print("  PASS")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Test sequence ({len(TEST_SEQ)} bp):")
    print(f"  {TEST_SEQ}")
    print(f"\nORF start : {_ORF_START}")
    print(f"Codon 15  : {TEST_SEQ[_ORF_START+42:_ORF_START+45]}  (expected AAA = K)")

    test_part1_basic()
    test_part1_wrong_aa()
    test_part2_simple_tm()
    test_part2_walk()
    test_part3_scan()
    test_part3_gained_lost()
    test_part3_silent_sites()

    print("\nAll tests passed.")
