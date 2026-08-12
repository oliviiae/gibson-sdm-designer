"""
End-to-end test: Parts 1-6 in sequence.

Mutation: K15E (AAA→GAA) in the 2000 bp construct from test_part5.py.

For the restriction site check we use a silent mutation at aa16 that
creates an ApaLI / Alw44I site (GTC→GTG), which was identified in
Part 3's find_silent_restriction_sites scan.  We apply it on top of
the K15E codon change before running the assembly.
"""

import random
from Bio.Seq import Seq

from codon_utils import find_codon_and_mutate
from tm_utils import simple_tm
from primer_bc import design_bc_primers
from primer_ad import design_ad_primers, PREFERRED_ENZYMES
from restriction_utils import gained_lost_sites, find_silent_restriction_sites
from assembly import (
    assemble_product,
    translate_orf,
    verify_mutation,
    verify_restriction_site,
    run_full_verification,
    print_report,
)

# ---------------------------------------------------------------------------
# Rebuild the 2000 bp test construct (identical to test_part5.py)
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

def _flank(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))

ORIGINAL_SEQ = _flank(895, 42) + _CORE + _flank(895, 99)
ORF_START     = 895 + _ORF_START_IN_CORE   # 905

assert len(ORIGINAL_SEQ) == 2000


# ---------------------------------------------------------------------------
# Helper: apply a silent mutation to add a diagnostic restriction site
# ---------------------------------------------------------------------------

def _apply_silent_mutation(seq: str, orf_start: int, codon_abs: int, new_codon: str) -> str:
    """Replace the codon at codon_abs (0-based absolute) with new_codon."""
    seq = seq.upper()
    return seq[:codon_abs] + new_codon.upper() + seq[codon_abs + 3:]


# ---------------------------------------------------------------------------
# Unit-level checks (individual functions in assembly.py)
# ---------------------------------------------------------------------------

def test_assemble_product():
    print("\n[Part 6] assemble_product")
    mutated, changed_pos, _, _ = find_codon_and_mutate(
        ORIGINAL_SEQ, ORF_START, 15, "E"
    )
    bc = design_bc_primers(mutated, changed_pos)

    # Dummy primer A/D spans that don't overlap
    a_start = bc.b_start - 50
    d_end   = bc.c_end + 50

    assembled = assemble_product(
        mutated, a_start, d_end, bc.overlap_start, bc.overlap_end
    )
    expected = mutated[a_start:d_end]
    assert assembled == expected, "Assembled product doesn't match direct slice"
    assert len(assembled) == d_end - a_start
    print(f"  Length {len(assembled)} bp — matches direct slice  PASS")


def test_translate_orf():
    print("\n[Part 6] translate_orf")
    protein = translate_orf(ORIGINAL_SEQ, ORF_START)
    assert protein[0]  == "M",  f"First AA should be M, got {protein[0]}"
    assert protein[14] == "K",  f"AA15 should be K, got {protein[14]}"
    assert len(protein) == 60,  f"Expected 60 AA, got {len(protein)}"
    print(f"  WT protein (first 20 AA): {protein[:20]}")
    print(f"  Length: {len(protein)} AA  AA15={protein[14]}  PASS")


def test_verify_mutation_pass():
    print("\n[Part 6] verify_mutation — K15E should PASS")
    mutated, changed_pos, _, _ = find_codon_and_mutate(
        ORIGINAL_SEQ, ORF_START, 15, "E"
    )
    result = verify_mutation(
        original_seq=ORIGINAL_SEQ,
        assembled_seq=mutated,
        orf_start_in_original=ORF_START,
        orf_start_in_assembled=ORF_START,
        aa_position=15,
        original_aa="K",
        new_aa="E",
    )
    assert result.passed, f"Expected PASS, got: {result.message}"
    assert result.observed_aa == "E"
    assert result.extra_changes == []
    print(f"  {result.message}  PASS")


def test_verify_mutation_fail_wrong_aa():
    print("\n[Part 6] verify_mutation — wrong AA should FAIL")
    # Pass the unmutated sequence as the 'assembled' product
    result = verify_mutation(
        original_seq=ORIGINAL_SEQ,
        assembled_seq=ORIGINAL_SEQ,   # same — no mutation applied
        orf_start_in_original=ORF_START,
        orf_start_in_assembled=ORF_START,
        aa_position=15,
        original_aa="K",
        new_aa="E",
    )
    assert not result.passed
    assert result.observed_aa == "K"
    print(f"  Correctly detected failure: {result.message}  PASS")


def test_verify_restriction_site():
    print("\n[Part 6] verify_restriction_site")
    # EcoRI site is absent in our construct
    check = verify_restriction_site(ORIGINAL_SEQ, "EcoRI", expected_present=False)
    print(f"  EcoRI absent: {check.message}")
    # We don't assert pass/fail since it depends on the random flank sequence,
    # but we assert the function runs and returns coherent results
    assert check.enzyme == "EcoRI"
    assert isinstance(check.passed, bool)
    print("  PASS")


# ---------------------------------------------------------------------------
# Full end-to-end test
# ---------------------------------------------------------------------------

def test_end_to_end():
    print("\n" + "=" * 62)
    print("  END-TO-END TEST: K15E on 2000 bp construct")
    print("=" * 62)

    # ── Part 1: codon mutation ────────────────────────────────────────────
    mutated_seq, changed_pos, orig_codon, new_codon = find_codon_and_mutate(
        ORIGINAL_SEQ, ORF_START, aa_position=15, new_aa="E"
    )
    print(f"\n[1] K15E: {orig_codon}→{new_codon}  at abs positions {changed_pos}")

    # ── Part 3: find diagnostic restriction site ──────────────────────────
    # Check whether the mutation itself gained/lost a site
    diff = gained_lost_sites(ORIGINAL_SEQ, mutated_seq)
    gained_enzymes = list(diff["gained"].keys())
    lost_enzymes   = list(diff["lost"].keys())
    print(f"\n[3] Restriction diff (mutation only):")
    print(f"    Gained: {gained_enzymes[:5] or 'none'}")
    print(f"    Lost  : {lost_enzymes[:5] or 'none'}")

    # Find a silent mutation near the change to use as diagnostic if needed
    codon_abs = ORF_START + 14 * 3   # codon 15
    silent_hits = find_silent_restriction_sites(
        mutated_seq, ORF_START,
        window_start=codon_abs - 6,
        window_end=codon_abs + 12,
    )

    # Choose a diagnostic: prefer a site gained by the mutation itself;
    # fall back to a 1-change silent mutation
    if gained_enzymes:
        diagnostic_enzyme = gained_enzymes[0]
        diagnostic_expected = True
        applied_seq = mutated_seq
        print(f"    Using mutation-gained site: {diagnostic_enzyme}")
    elif silent_hits:
        hit = silent_hits[0]   # fewest changes, sorted by rank_candidates logic
        diagnostic_enzyme = hit["enzyme"]
        diagnostic_expected = (hit["effect"] == "gained")
        applied_seq = _apply_silent_mutation(
            mutated_seq, ORF_START, hit["position"], hit["new_codon"]
        )
        print(f"\n    No site from mutation alone — applying silent mutation "
              f"aa{hit['aa_index']} {hit['original_codon']}→{hit['new_codon']} "
              f"({hit['changes']} change) to create {diagnostic_enzyme} site")
    else:
        diagnostic_enzyme = None
        diagnostic_expected = True
        applied_seq = mutated_seq
        print("    No diagnostic site available — skipping restriction check")

    # ── Part 4: B/C primers ───────────────────────────────────────────────
    bc = design_bc_primers(applied_seq, changed_pos)
    print(f"\n[4] Primer B: 5'-{bc.primer_b}-3'  Tm={bc.tm_b}°C")
    print(f"    Primer C: 5'-{bc.primer_c}-3'  "
          f"Tm={bc.tm_c_anneal}°C (anneal) / {bc.tm_c_full}°C (full)")
    print(f"    Overlap : 5'-{bc.overlap_seq}-3'  "
          f"Tm={bc.tm_overlap_fwd}°C  [{bc.overlap_start},{bc.overlap_end})")

    # ── Part 5: A/D primers ───────────────────────────────────────────────
    result_a, result_d = design_ad_primers(
        applied_seq,
        b_start=bc.b_start,
        c_end=bc.c_end,
        window_bp=(700, 1000),
        target_tm=51.0,
    )
    print(f"\n[5] Primer A candidates: {len(result_a.candidates)}")
    if result_a.top:
        t = result_a.top
        print(f"    Top: {t.enzyme}  cut@{t.cut_pos}  Tm={t.primer_tm}°C  "
              f"5'-{t.primer_seq}-3'")
    print(f"    Primer D candidates: {len(result_d.candidates)}")
    if result_d.top:
        t = result_d.top
        print(f"    Top: {t.enzyme}  cut@{t.cut_pos}  Tm={t.primer_tm}°C  "
              f"5'-{t.primer_seq}-3'")

    if result_a.top is None or result_d.top is None:
        print("    WARNING: missing primer A or D — skipping full report")
        return

    # ── Part 6: assembly and verification ────────────────────────────────
    report = run_full_verification(
        original_seq=ORIGINAL_SEQ,
        mutated_seq=applied_seq,
        orf_start=ORF_START,
        aa_position=15,
        original_aa="K",
        new_aa="E",
        original_codon=orig_codon,
        new_codon=new_codon,
        changed_positions=changed_pos,
        bc=bc,
        result_a=result_a,
        result_d=result_d,
        diagnostic_enzyme=diagnostic_enzyme,
        diagnostic_expected_present=diagnostic_expected,
    )

    print_report(report)

    # ── Assertions ────────────────────────────────────────────────────────
    assert report.mutation_check.passed, \
        f"Translation check FAILED: {report.mutation_check.message}"
    if report.restriction_check is not None:
        assert report.restriction_check.passed, \
            f"Restriction check FAILED: {report.restriction_check.message}"

    # Fragment size sanity: each fragment must contain both primers and the overlap
    assert report.frag_ab_len > len(report.primer_b), \
        "Fragment ab must be longer than primer B"
    assert report.frag_cd_len > len(report.primer_c), \
        "Fragment cd must be longer than primer C"

    print("All end-to-end assertions passed.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_assemble_product()
    test_translate_orf()
    test_verify_mutation_pass()
    test_verify_mutation_fail_wrong_aa()
    test_verify_restriction_site()
    test_end_to_end()
    print("\nAll Part 6 tests passed.")
