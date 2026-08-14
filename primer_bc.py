"""
Part 4 — Primer B / C boundaries and overlap region.

Public API
----------
find_bc_overlap(seq, overlap_start_pos, min_len, tm_range)
    Scan forward from overlap_start_pos for the shortest window (≥ min_len) whose
    Tm sits in tm_range on both the forward strand and its reverse complement.
    (With the Wallace sum rule the two values are always identical because RC
    preserves base composition.  Both are checked explicitly so this function
    remains correct if the Tm rule is ever upgraded.)

design_bc_primers(mutated_seq, changed_positions, tm_range, min_overlap)
    Full Part-4 pipeline.  Returns a PrimerBCResult dataclass.

PrimerBCResult fields
---------------------
b_start, b_end      0-based half-open indices of primer B in mutated_seq
c_start, c_end      0-based half-open indices of primer C template strand
primer_b            sequence of primer B  (5'→3', sense strand)
primer_c            sequence of primer C  (5'→3', antisense — ready to order)
primer_c_fwd        template-strand sequence of primer C region (diagnostic)
overlap_seq         the shared sequence
overlap_start       0-based start of overlap in mutated_seq
overlap_end         0-based end   of overlap in mutated_seq (exclusive)
tm_b                Wallace Tm of primer B. Usually == overlap Tm, since the
                    overlap search tries starting exactly at b_start first —
                    but may differ slightly if the overlap had to start a
                    few bases later to satisfy the G/C-terminus + Tm-range
                    constraints together (primer B's start never moves).
tm_c_full           Wallace Tm of full primer C  (overlap + downstream annealing)
tm_c_anneal         Wallace Tm of primer C's unique downstream annealing region only
                    [overlap_end, c_end) — the portion that does NOT appear in primer B
tm_overlap_fwd      Wallace Tm of overlap on forward strand
tm_overlap_rc       Wallace Tm of overlap on reverse strand (always == fwd)
"""

from dataclasses import dataclass
from Bio.Seq import Seq

from tm_utils import simple_tm, walk_to_tm


@dataclass
class PrimerBCResult:
    # Primer B (forward, sense strand)
    b_start: int
    b_end: int
    primer_b: str
    tm_b: float

    # Primer C (reverse; primer_c is the sequence you order, 5'→3' antisense)
    c_start: int          # start on forward strand (= overlap_start)
    c_end: int            # end on forward strand (exclusive)
    primer_c: str         # reverse complement — the actual oligo
    primer_c_fwd: str     # forward-strand span (for position reference)
    tm_c_full: float      # Tm of the complete primer C (overlap + annealing tail)
    tm_c_anneal: float    # Tm of the unique annealing region [overlap_end, c_end)

    # Overlap
    overlap_start: int
    overlap_end: int
    overlap_seq: str
    tm_overlap_fwd: float
    tm_overlap_rc: float


# ---------------------------------------------------------------------------

def find_bc_overlap(
    seq: str,
    search_from: int,
    must_cover: int | None = None,
    min_len: int = 16,
    tm_range: tuple[float, float] = (48.0, 54.0),
    max_search: int = 120,
    max_start_shift: int = 40,
) -> tuple[str, int, int, float, float]:
    """
    Find the overlap region shared by primers B and C.

    Per protocol: the overlap must (a) start AND end on a G/C base,
    (b) have Tm in tm_range on both strands, (c) be ≥ min_len, and
    (d) cover the mutated base(s) (must_cover = the last mutated position,
    0-based — the window's end must extend past it).

    The overlap's start does not have to be exactly search_from (= primer
    B's start) — only its *end* has to match primer B's end. So this tries
    start = search_from first (the common case, since search_from already
    landed on a G/C base by construction), then slides the start forward one
    base at a time (up to max_start_shift bases) if no valid end can be
    found from that start, since the G/C-start requirement plus the
    G/C-end + Tm-range requirement together can be unsatisfiable from a
    single fixed start position.

    Returns (overlap_seq, start, end, tm_fwd, tm_rc).
    Raises RuntimeError if no qualifying window is found.

    Note: with the Wallace sum rule Tm(forward) == Tm(RC) for any sequence,
    because reverse-complementation swaps A↔T and G↔C, leaving base counts
    unchanged. Both are still checked for forward-compatibility. Likewise,
    reverse-complementing swaps and reverses the terminal bases, so a window
    whose forward strand starts and ends in G/C automatically satisfies the
    same requirement on the RC strand — no separate check needed there.
    """
    seq = seq.upper()
    tm_lo, tm_hi = tm_range

    for start in range(search_from, search_from + max_start_shift + 1):
        if start >= len(seq) or seq[start] not in "GC":
            continue
        for length in range(min_len, min_len + max_search):
            end = start + length
            if end > len(seq):
                break
            if must_cover is not None and end <= must_cover:
                continue   # doesn't cover the mutation yet
            window = seq[start:end]
            if window[-1] not in "GC":
                continue
            tm_fwd = simple_tm(window)
            tm_rc  = simple_tm(str(Seq(window).reverse_complement()))
            if tm_lo <= tm_fwd <= tm_hi and tm_lo <= tm_rc <= tm_hi:
                return window, start, end, tm_fwd, tm_rc

    raise RuntimeError(
        f"No overlap window ≥{min_len}bp with Tm in {tm_range}, G/C at both "
        f"ends, and covering position {must_cover} found starting near "
        f"position {search_from}."
    )


def design_bc_primers(
    mutated_seq: str,
    changed_positions: list[int],
    tm_range: tuple[float, float] = (48.0, 54.0),
    min_overlap: int = 16,
) -> PrimerBCResult:
    """
    Design primers B and C for Gibson SDM.

    Parameters
    ----------
    mutated_seq       Full mutated DNA sequence (sense strand, 5'→3').
    changed_positions 0-based absolute positions of mutated nucleotides
                      (from find_codon_and_mutate).
    tm_range          Target Tm window in °C (Wallace rule).
    min_overlap       Minimum overlap length in bp (default 16).

    Design logic
    ------------
    1. Walk 5' from the first mutated base to find primer B's start:
       accumulate Tm leftward until in tm_range AND on a G/C.

    2. Walk 3' from the last mutated base to find primer C's end:
       accumulate Tm rightward until in tm_range AND on a G/C.

    3. Find the overlap: window ≥ min_overlap, starting on a G/C at or after
       b_start and ending on a G/C, covering the mutation, whose forward-
       strand AND reverse-complement Tm are both in tm_range.

    4. Primer B = mutated_seq[b_start : overlap_end]  (sense, 5'→3')
       Primer C = RC of mutated_seq[overlap_start : c_end]  (antisense, 5'→3')
    """
    mutated_seq = mutated_seq.upper()
    first_mut = min(changed_positions)
    last_mut  = max(changed_positions)

    # --- Step 1: B start (walk 5') -------------------------------------------
    # Protocol: "starting right before the first change... upstream" — the
    # walk's anchor is the base immediately 5' of the first mutated
    # nucleotide, not the mutated base itself.
    b_anchor = max(0, first_mut - 1)
    _, b_start, _, _ = walk_to_tm(
        mutated_seq,
        anchor_pos=b_anchor,
        direction=-1,
        tm_range=tm_range,
        must_end_gc=True,
    )

    # --- Step 2: C end (walk 3') ----------------------------------------------
    # Protocol: "starting right after the last change... downstream" — the
    # walk's anchor is the base immediately 3' of the last mutated
    # nucleotide, not the mutated base itself.
    c_anchor = min(len(mutated_seq) - 1, last_mut + 1)
    _, _, c_end, _ = walk_to_tm(
        mutated_seq,
        anchor_pos=c_anchor,
        direction=1,
        tm_range=tm_range,
        must_end_gc=True,
    )

    # --- Step 3: overlap region -----------------------------------------------
    # The overlap must start on a G/C at or after b_start, end on a G/C,
    # cover the mutation (through last_mut), and have Tm in range on both
    # strands. Its start need not be exactly b_start — only its end has to
    # coincide with primer B's end.
    overlap_seq, ov_start, ov_end, tm_ov_fwd, tm_ov_rc = find_bc_overlap(
        mutated_seq,
        search_from=b_start,
        must_cover=last_mut,
        min_len=min_overlap,
        tm_range=tm_range,
    )

    # c_end was walked independently, before the overlap was known. If the
    # overlap ends up extending as far as (or past) c_end — which a larger
    # min_overlap makes more likely — primer C would have zero unique
    # annealing bases beyond the shared overlap: it wouldn't actually bind
    # fresh template at all, and couldn't function as a real primer. Treat
    # that the same as any other unsatisfiable design, not a silent Tm=0.
    if ov_end >= c_end:
        raise RuntimeError(
            f"Overlap end (position {ov_end}) reached or passed primer C's "
            f"end (position {c_end}), leaving no unique annealing region for "
            f"primer C. Try a smaller --min-overlap."
        )

    # --- Step 4: build primer sequences --------------------------------------
    primer_b     = mutated_seq[b_start:ov_end]          # forward, sense
    primer_c_fwd = mutated_seq[ov_start:c_end]           # template region
    primer_c     = str(Seq(primer_c_fwd).reverse_complement())

    return PrimerBCResult(
        b_start=b_start,
        b_end=ov_end,
        primer_b=primer_b,
        tm_b=simple_tm(primer_b),

        c_start=ov_start,
        c_end=c_end,
        primer_c=primer_c,
        primer_c_fwd=primer_c_fwd,
        tm_c_full=simple_tm(primer_c_fwd),
        tm_c_anneal=simple_tm(mutated_seq[ov_end:c_end]),  # downstream-only portion

        overlap_start=ov_start,
        overlap_end=ov_end,
        overlap_seq=overlap_seq,
        tm_overlap_fwd=tm_ov_fwd,
        tm_overlap_rc=tm_ov_rc,
    )
