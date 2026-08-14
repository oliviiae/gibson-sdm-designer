"""
Part 6 — Gibson assembly reconstruction and end-to-end verification.

Public API
----------
assemble_product(mutated_seq, primer_a_start, primer_d_end, overlap_start, overlap_end)
    Reconstruct the linear PCR product that results from Gibson assembly of
    fragments ab and cd.

translate_orf(seq, orf_start, orf_start_in_seq)
    Translate an ORF within seq.  Stops at the first in-frame stop codon.

verify_mutation(original_seq, assembled_seq, orf_start_abs, orf_start_in_assembled,
                aa_position, original_aa, new_aa)
    Compare translated ORFs, confirm the target substitution occurred, and
    confirm no other amino acids changed.

verify_restriction_site(seq, enzyme_name, expected_present)
    Check whether a named restriction enzyme cuts in seq and whether that
    matches the expectation.

run_full_verification(...)
    Runs all checks and returns a VerificationReport dataclass.

print_report(report)
    Pretty-print the report to stdout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from Bio.Seq import Seq
from Bio.Restriction import RestrictionBatch, Analysis, CommOnly


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_product(
    mutated_seq: str,
    primer_a_start: int,
    primer_d_end: int,
    overlap_start: int,
    overlap_end: int,
) -> str:
    """
    Reconstruct the Gibson-assembled linear product.

    The two PCR fragments are:
      Fragment ab : mutated_seq[primer_a_start : overlap_end]
      Fragment cd : mutated_seq[overlap_start  : primer_d_end]

    They share the overlap window [overlap_start, overlap_end).
    Gibson exonuclease chews back both ends to expose complementary
    single-stranded overhangs; after annealing and fill-in the
    overlap appears exactly once.

    Assembled = fragment_ab + fragment_cd[overlap_len:]
              = mutated_seq[primer_a_start : primer_d_end]

    Returns the assembled sequence string (5'→3', sense strand).
    """
    mutated_seq = mutated_seq.upper()
    overlap_len = overlap_end - overlap_start

    frag_ab = mutated_seq[primer_a_start:overlap_end]
    frag_cd = mutated_seq[overlap_start:primer_d_end]

    # Sanity: the tail of ab == the head of cd
    if frag_ab[-overlap_len:] != frag_cd[:overlap_len]:
        raise ValueError(
            "Overlap mismatch: the last overlap_len bases of fragment ab "
            "do not equal the first overlap_len bases of fragment cd.\n"
            f"  ab tail : {frag_ab[-overlap_len:]}\n"
            f"  cd head : {frag_cd[:overlap_len]}"
        )

    assembled = frag_ab + frag_cd[overlap_len:]
    # Equivalent to mutated_seq[primer_a_start:primer_d_end]
    assert assembled == mutated_seq[primer_a_start:primer_d_end], \
        "Assembly result inconsistent with direct slice — logic error"
    return assembled


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate_orf(seq: str, orf_start: int) -> str:
    """
    Translate the ORF that begins at orf_start (0-based index within seq).
    Stops at — and excludes — the first in-frame stop codon.
    Raises ValueError if orf_start is out of range.
    """
    seq = seq.upper()
    if orf_start < 0:
        raise ValueError(f"orf_start {orf_start} is negative")
    if orf_start >= len(seq):
        raise ValueError(f"orf_start {orf_start} is beyond sequence length {len(seq)}")
    orf_seq = seq[orf_start:]
    # Trim to complete codons
    orf_seq = orf_seq[: (len(orf_seq) // 3) * 3]
    protein = str(Seq(orf_seq).translate())
    # Truncate at first stop codon
    stop = protein.find("*")
    return protein[:stop] if stop != -1 else protein


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

@dataclass
class MutationCheck:
    passed: bool
    aa_position: int          # 1-based (first mutation, for backward compat)
    expected_aa: str          # (first mutation, for backward compat)
    observed_aa: str          # (first mutation, for backward compat)
    extra_changes: list[tuple[int, str, str]] = field(default_factory=list)
    # extra_changes: list of (1-based position, original_aa, new_aa)
    message: str = ""
    # per_mutation: full detail for every intended mutation (length 1 for a
    # single-mutation check, length N for a combined multi-mutation check).
    # Each entry: (1-based position, expected_aa, observed_aa, passed)
    per_mutation: list[tuple[int, str, str, bool]] = field(default_factory=list)


@dataclass
class RestrictionCheck:
    passed: bool
    enzyme: str
    expected_present: bool
    observed_present: bool
    cut_positions: list[int] = field(default_factory=list)   # 1-based, in assembled seq
    message: str = ""


@dataclass
class VerificationReport:
    # Input summary
    mutation_label: str          # e.g. "K15E"
    original_codon: str
    new_codon: str
    changed_positions: list[int] # absolute positions in full seq

    # Primer sequences (5'→3', ready to order)
    primer_a: str
    primer_b: str
    primer_c: str                # antisense
    primer_d: str
    primer_a_tm: float
    primer_b_tm: float
    primer_c_tm_full: float
    primer_c_tm_anneal: float
    primer_d_tm: float
    primer_a_enzyme: str
    primer_d_enzyme: str

    # Overlap
    overlap_seq: str
    overlap_tm: float

    # Fragment sizes
    frag_ab_len: int
    frag_cd_len: int
    assembled_len: int

    # Fragment sequences (5'→3', sense strand)
    frag_ab_seq: str
    frag_cd_seq: str
    assembled_seq: str

    # Verification
    mutation_check: MutationCheck
    restriction_check: RestrictionCheck | None   # None if no diagnostic site


def verify_mutation(
    original_seq: str,
    assembled_seq: str,
    orf_start_in_original: int,
    orf_start_in_assembled: int,
    aa_position: int | list[int],
    original_aa: str | list[str],
    new_aa: str | list[str],
) -> MutationCheck:
    """
    Translate both sequences and compare their proteins.

    aa_position/original_aa/new_aa each accept either a single scalar (one
    mutation) or equal-length lists (a combined multi-mutation design, e.g.
    a double mutation sharing one primer set). original_aa is accepted for
    API symmetry with the rest of the pipeline but isn't itself used here —
    the WT protein already encodes it, so it's implicitly checked via the
    "no unexpected changes" scan below.

    Returns a MutationCheck with:
      - passed = True iff every position changed to its expected new_aa AND
        no other positions differ
      - extra_changes = list of (pos, old_aa, new_aa) for unexpected differences
      - per_mutation = per-position (pos, expected, observed, passed) detail
    """
    positions = list(aa_position) if isinstance(aa_position, (list, tuple)) else [aa_position]
    new_aas   = list(new_aa)      if isinstance(new_aa, (list, tuple))      else [new_aa]
    if len(positions) != len(new_aas):
        raise ValueError(
            f"aa_position and new_aa must be the same length "
            f"({len(positions)} vs {len(new_aas)})."
        )
    new_aas = [a.upper() for a in new_aas]

    wt_protein  = translate_orf(original_seq,  orf_start_in_original)
    mut_protein = translate_orf(assembled_seq, orf_start_in_assembled)

    per_mutation: list[tuple[int, str, str, bool]] = []
    for pos, exp_aa in zip(positions, new_aas):
        aa_idx = pos - 1   # 0-based
        if exp_aa == "*":
            # A nonsense (stop-codon) mutation is verified differently:
            # translate_orf always strips the stop codon itself from its
            # return value, so mut_protein can never literally contain '*'
            # at aa_idx — the correct signal instead is that translation
            # terminates exactly one residue short of this position.
            if len(mut_protein) == aa_idx:
                observed, ok = "*", True
            elif len(mut_protein) > aa_idx:
                observed, ok = mut_protein[aa_idx], False
            else:
                observed, ok = "?", False
        else:
            if aa_idx >= len(mut_protein):
                observed, ok = "?", False
            else:
                observed = mut_protein[aa_idx]
                ok = observed == exp_aa
        per_mutation.append((pos, exp_aa, observed, ok))

    target_ok = all(ok for _, _, _, ok in per_mutation)
    mutated_positions = {pos for pos, _, _, _ in per_mutation}

    # Scan for any other differences, over the region actually present in
    # both proteins — for a stop mutation, mut_protein simply ends earlier,
    # so this naturally limits the comparison to before the truncation point.
    compare_len = min(len(wt_protein), len(mut_protein))
    extra: list[tuple[int, str, str]] = []
    for i in range(1, compare_len + 1):
        if i in mutated_positions:
            continue
        wt, mt = wt_protein[i - 1], mut_protein[i - 1]
        if wt != mt:
            extra.append((i, wt, mt))

    passed = target_ok and len(extra) == 0
    parts = []
    for pos, exp_aa, observed, ok in per_mutation:
        if not ok:
            parts.append(f"Position {pos}: expected {exp_aa}, observed {observed}")
    if extra:
        parts.append(
            "Unexpected AA changes: " +
            ", ".join(f"{wt}{pos}{mt}" for pos, wt, mt in extra)
        )

    first_pos, first_exp, first_obs, _ = per_mutation[0]
    return MutationCheck(
        passed=passed,
        aa_position=first_pos,
        expected_aa=first_exp,
        observed_aa=first_obs,
        extra_changes=extra,
        message="OK" if passed else "; ".join(parts),
        per_mutation=per_mutation,
    )


def verify_restriction_site(
    seq: str,
    enzyme_name: str,
    expected_present: bool,
) -> RestrictionCheck:
    """
    Check whether enzyme_name cuts seq at least once.
    Returns a RestrictionCheck whose .passed is True when the observed
    presence matches expected_present.
    """
    from Bio.Restriction import AllEnzymes
    seq = seq.upper()

    try:
        enz = AllEnzymes.get(enzyme_name)
        if enz is None:
            raise KeyError(enzyme_name)
    except (KeyError, AttributeError):
        return RestrictionCheck(
            passed=False,
            enzyme=enzyme_name,
            expected_present=expected_present,
            observed_present=False,
            message=f"Unknown enzyme '{enzyme_name}'",
        )

    rb = RestrictionBatch([enz])
    analysis = Analysis(rb, Seq(seq), linear=True)
    cuts = analysis.full().get(enz, [])
    observed_present = len(cuts) > 0
    passed = observed_present == expected_present

    if passed:
        if expected_present:
            msg = f"{enzyme_name} cuts at position(s) {cuts} — PRESENT as expected"
        else:
            msg = f"{enzyme_name} does not cut — ABSENT as expected"
    else:
        if expected_present:
            msg = f"{enzyme_name} expected to cut but NO site found"
        else:
            msg = f"{enzyme_name} expected to be absent but cuts at {cuts}"

    return RestrictionCheck(
        passed=passed,
        enzyme=enzyme_name,
        expected_present=expected_present,
        observed_present=observed_present,
        cut_positions=list(cuts),
        message=msg,
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_verification(
    original_seq: str,
    mutated_seq: str,
    orf_start: int,
    aa_position: int | list[int],
    original_aa: str | list[str],
    new_aa: str | list[str],
    original_codon: str,
    new_codon: str,
    changed_positions: list[int],
    bc,             # PrimerBCResult from primer_bc
    result_a,       # FlankingPrimerResult from primer_ad
    result_d,       # FlankingPrimerResult from primer_ad
    diagnostic_enzyme: str | None = None,
    diagnostic_expected_present: bool = True,
    mutation_label: str | None = None,
) -> VerificationReport:
    """
    Run the full Part-6 verification pipeline.

    aa_position/original_aa/new_aa accept either scalars (one mutation) or
    equal-length lists (a combined multi-mutation design).

    Parameters
    ----------
    diagnostic_enzyme          : enzyme to check in the assembled product.
                                 None = skip restriction check.
    diagnostic_expected_present: True if the site should be GAINED by the
                                 mutation (present in assembled, absent in WT);
                                 False if it should be LOST.
    mutation_label              : precomputed display label; if omitted, one
                                 is built from original_aa+aa_position+new_aa
                                 (joined with " + " for multiple mutations).
    """
    top_a = result_a.top
    top_d = result_d.top

    if top_a is None:
        raise ValueError("No primer A candidates — widen the search window or use a longer sequence.")
    if top_d is None:
        raise ValueError("No primer D candidates — widen the search window or use a longer sequence.")

    # --- Assemble ------------------------------------------------------------
    assembled = assemble_product(
        mutated_seq,
        primer_a_start=top_a.primer_start,
        primer_d_end=top_d.primer_end,
        overlap_start=bc.overlap_start,
        overlap_end=bc.overlap_end,
    )

    # Fragment sizes and sequences (each fragment includes the overlap once)
    frag_ab_len = bc.overlap_end   - top_a.primer_start
    frag_cd_len = top_d.primer_end - bc.overlap_start
    frag_ab_seq = mutated_seq[top_a.primer_start:bc.overlap_end]
    frag_cd_seq = mutated_seq[bc.overlap_start:top_d.primer_end]

    # --- Translation check ---------------------------------------------------
    # Translate the FULL construct (not just the isolated A→D PCR fragment).
    # The A→D fragment is only the piece actually synthesized/amplified — for
    # any real gene it will rarely contain the ORF's start codon (primer A is
    # typically hundreds of bp downstream of ATG), so translating it in
    # isolation with orf_start adjusted to fragment-local coordinates goes
    # negative and silently translates garbage from the wrong end of the
    # string. The mutation and its correctness are still fully verifiable
    # from the full mutated_seq, which already has the codon change applied.
    mut_check = verify_mutation(
        original_seq=original_seq,
        assembled_seq=mutated_seq,
        orf_start_in_original=orf_start,
        orf_start_in_assembled=orf_start,
        aa_position=aa_position,
        original_aa=original_aa,
        new_aa=new_aa,
    )

    # --- Restriction check ---------------------------------------------------
    rest_check: RestrictionCheck | None = None
    if diagnostic_enzyme:
        rest_check = verify_restriction_site(
            assembled, diagnostic_enzyme, diagnostic_expected_present
        )

    if mutation_label is None:
        _positions = aa_position if isinstance(aa_position, (list, tuple)) else [aa_position]
        _origs     = original_aa if isinstance(original_aa, (list, tuple)) else [original_aa]
        _news      = new_aa      if isinstance(new_aa, (list, tuple))      else [new_aa]
        mutation_label = " + ".join(
            f"{oa}{p}{na}" for p, oa, na in zip(_positions, _origs, _news)
        )

    return VerificationReport(
        mutation_label=mutation_label,
        original_codon=original_codon,
        new_codon=new_codon,
        changed_positions=changed_positions,

        primer_a=top_a.primer_seq,
        primer_b=bc.primer_b,
        primer_c=bc.primer_c,
        primer_d=top_d.primer_seq,
        primer_a_tm=top_a.primer_tm,
        primer_b_tm=bc.tm_b,
        primer_c_tm_full=bc.tm_c_full,
        primer_c_tm_anneal=bc.tm_c_anneal,
        primer_d_tm=top_d.primer_tm,
        primer_a_enzyme=top_a.enzyme,
        primer_d_enzyme=top_d.enzyme,

        overlap_seq=bc.overlap_seq,
        overlap_tm=bc.tm_overlap_fwd,

        frag_ab_len=frag_ab_len,
        frag_cd_len=frag_cd_len,
        assembled_len=len(assembled),
        frag_ab_seq=frag_ab_seq,
        frag_cd_seq=frag_cd_seq,
        assembled_seq=assembled,

        mutation_check=mut_check,
        restriction_check=rest_check,
    )


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_report(report: VerificationReport) -> None:
    W = 62
    sep  = "─" * W
    dsep = "═" * W

    def _status(passed: bool) -> str:
        return "PASS ✓" if passed else "FAIL ✗"

    print(f"\n{'═'*W}")
    print(f"  Gibson Assembly Report — {report.mutation_label}")
    print(f"{'═'*W}")

    print(f"\n  Mutation")
    print(f"    {report.mutation_label}  "
          f"({report.original_codon} → {report.new_codon})")
    print(f"    Changed nucleotide(s): {report.changed_positions}")

    print(f"\n  Primers  (5'→3')")
    print(f"    A  [{report.primer_a_enzyme:<8}]  Tm={report.primer_a_tm:>3.0f}°C  "
          f"{report.primer_a}")
    print(f"    B  [overlap  ]  Tm={report.primer_b_tm:>3.0f}°C  {report.primer_b}")
    print(f"    C  [overlap  ]  Tm={report.primer_c_tm_anneal:>3.0f}°C (anneal) / "
          f"{report.primer_c_tm_full:.0f}°C (full)")
    print(f"                         {report.primer_c}  (antisense)")
    print(f"    D  [{report.primer_d_enzyme:<8}]  Tm={report.primer_d_tm:>3.0f}°C  "
          f"{report.primer_d}")

    print(f"\n  Overlap")
    print(f"    {report.overlap_seq}  ({len(report.overlap_seq)} bp, "
          f"Tm={report.overlap_tm:.0f}°C)")

    print(f"\n  Predicted PCR products")
    print(f"    Fragment ab  {report.frag_ab_len:>6} bp  "
          f"(primer A start → overlap end)")
    print(f"    {report.frag_ab_seq}")
    print(f"    Fragment cd  {report.frag_cd_len:>6} bp  "
          f"(overlap start → primer D end)")
    print(f"    {report.frag_cd_seq}")
    print(f"    Assembled    {report.assembled_len:>6} bp")
    print(f"    {report.assembled_seq}")

    print(f"\n  {sep}")
    mc = report.mutation_check
    status = _status(mc.passed)
    print(f"  Translation check          {status:>10}")
    for pos, exp_aa, observed, ok in mc.per_mutation:
        print(f"    Position {pos}: {observed} observed  "
              f"(expected {exp_aa}){'' if ok else '  ✗ MISMATCH'}")
    if mc.extra_changes:
        print(f"    Unexpected changes:")
        for pos, wt, mt in mc.extra_changes:
            print(f"      {wt}{pos}{mt}")
    else:
        print(f"    No other amino acid changes")
    if not mc.passed:
        print(f"    Detail: {mc.message}")

    rc = report.restriction_check
    if rc is not None:
        print(f"\n  Restriction site check     {_status(rc.passed):>10}")
        print(f"    {rc.message}")
    else:
        print(f"\n  Restriction site check      (skipped — no diagnostic site provided)")

    overall = mc.passed and (rc is None or rc.passed)
    print(f"\n{'═'*W}")
    print(f"  Overall result             {_status(overall):>10}")
    print(f"{'═'*W}\n")
