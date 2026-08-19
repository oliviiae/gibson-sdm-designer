"""
Part 5 — Flanking primers A and D.

Public API
----------
find_unique_sites_in_window(full_seq, window_start, window_end)
    Scan the full construct for all 6-8 bp commercial restriction sites,
    keep only those that cut exactly once in the entire construct, then
    return the subset whose cut position falls inside [window_start, window_end).

design_flanking_primer(seq, cut_pos, direction, tm_range, min_len)
    Starting at cut_pos, walk in direction toward the mutation, accumulating
    Wallace Tm, until in [48-54°C] on a G/C base.  Returns primer sequence,
    genomic span, and Tm.

rank_candidates(candidates, target_tm)
    Sort a list of candidate dicts by |Tm - target_tm| ascending, breaking
    ties by enzyme name for determinism.

design_ad_primers(full_seq, b_start, c_end, tm_range, window_bp, min_len)
    Full Part-5 pipeline.  Returns a FlankingPrimerResult dataclass for each
    of primers A and D.

Window geometry
---------------
Primer A search window: [b_start - window_bp[1], b_start - window_bp[0])
    e.g. window_bp=(700,1000) → 300 bp band, 700-1000 bp upstream of b_start
Primer D search window: [c_end + window_bp[0], c_end + window_bp[1])
    e.g. window_bp=(700,1000) → 300 bp band, 700-1000 bp downstream of c_end

For primer A a unique site at position p means:
    primer walks rightward (+1) from p toward the mutation.
For primer D a unique site at position p means:
    primer walks leftward  (-1) from p toward the mutation.
"""

from dataclasses import dataclass, field
from Bio.Restriction import RestrictionBatch, Analysis, CommOnly
from Bio.Seq import Seq

from tm_utils import simple_tm, walk_to_tm


# ---------------------------------------------------------------------------
# Restriction batch (same 6-8 bp commercial subset as Part 3)
# ---------------------------------------------------------------------------

def _make_batch() -> RestrictionBatch:
    rb = RestrictionBatch(CommOnly)
    return RestrictionBatch([e for e in rb if 6 <= len(e.site) <= 8])


_BATCH = _make_batch()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CandidateSite:
    enzyme: str          # enzyme name string
    cut_pos: int         # 1-based cut position in the full sequence
    cut_pos_0: int       # 0-based, for direct use as slice index
    primer_seq: str      # designed primer (5'→3')
    primer_start: int    # 0-based start in full_seq (inclusive)
    primer_end: int      # 0-based end   in full_seq (exclusive)
    primer_tm: float     # Wallace Tm of the designed primer
    tm_dist: float       # |primer_tm - target_tm|
    extended_tm: bool = False   # True if Tm is above tm_range[1] (allowed up
                                 # to extended_tm_max only because the primer's
                                 # 3' base is A or T)


@dataclass
class FlankingPrimerResult:
    label: str                         # 'A' or 'D'
    window_start: int                  # search window coords (0-based)
    window_end: int
    candidates: list[CandidateSite] = field(default_factory=list)

    @property
    def top(self) -> CandidateSite | None:
        return self.candidates[0] if self.candidates else None


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def find_unique_sites_in_window(
    full_seq: str,
    window_start: int,
    window_end: int,
) -> dict[str, list[int]]:
    """
    Return restriction sites that are (a) unique in full_seq (cut exactly once)
    and (b) whose 1-based cut position converts to a 0-based index that falls
    in [window_start, window_end).

    Returns
    -------
    dict mapping enzyme name → list of 0-based cut positions within the window.
    (List will always be length 1 for unique cutters, but the type is kept
    consistent with the rest of the restriction_utils API.)
    """
    full_seq = full_seq.upper()
    analysis = Analysis(_BATCH, Seq(full_seq), linear=True)
    raw = analysis.full()

    result: dict[str, list[int]] = {}
    for enz, positions in raw.items():
        if len(positions) != 1:
            continue                         # not unique in the construct
        cut_1based = positions[0]
        cut_0based = cut_1based - 1          # convert to 0-based
        if window_start <= cut_0based < window_end:
            result[str(enz)] = [cut_0based]

    return result


def design_flanking_primer(
    seq: str,
    cut_pos: int,
    direction: int,
    tm_range: tuple[float, float] = (48.0, 54.0),
    min_len: int = 16,
    max_len: int = 100,
    extended_tm_max: float = 56.0,
) -> tuple[str, int, int, float, bool]:
    """
    Design a flanking primer anchored at cut_pos, walking in direction toward
    the mutation.

    Parameters
    ----------
    seq       : full DNA sequence
    cut_pos   : 0-based anchor position (the restriction cut site)
    direction : +1 for primer A (walk rightward toward mutation)
                -1 for primer D (walk leftward toward mutation)
    tm_range  : target Wallace Tm window
    min_len   : minimum primer length (protocol: >=16nt overlap with vector)
    max_len   : safety cap on search length
    extended_tm_max : a Tm above tm_range[1] (up to this ceiling) is still
                accepted, but only if the primer's 3' base is A or T —
                used as a fallback when no length hits the normal window.

    Per protocol, A/D need Tm 48-54°C AND >=16nt — both at once, not one
    then the other. Searching length-by-length starting at min_len (rather
    than walking to a Tm-satisfying stop point and then force-extending
    for length) avoids a case where reaching min_len only after Tm was
    already satisfied at a shorter length pushes the Tm back out of range.
    Unlike primers B/C, there is no G/C-ending requirement for A/D in the
    protocol, so none is enforced here — except in the extended-Tm fallback,
    where an A/T 3' end is what makes the higher Tm acceptable.

    Orientation: primer A anneals to the template as the forward primer, so
    its oligo is the template-strand span unmodified. Primer D sits at the
    far (downstream) end of fragment "cd" and must close that PCR product
    back toward primer C, so its real oligo is the reverse complement of
    the template-strand span — matching primer B's role relative to A (see
    primer_bc.py). The A/T-terminus check for the extended-Tm fallback is
    applied to this final, correctly-oriented oligo's actual 3' base, not
    the template-strand span's last character (which, for primer D, is not
    the same base after reverse-complementing).

    Returns
    -------
    (primer_seq, start, end, tm, extended_tm) if a length in [min_len, max_len]
    satisfies the length requirement and either lands Tm in the normal
    48-54°C window (extended_tm=False), or lands Tm in (54, extended_tm_max]
    with a 3' A/T base (extended_tm=True). primer_seq is the actual oligo to
    order — reverse-complemented already for primer D. start/end always
    describe the template-strand span (needed for fragment position math),
    regardless of primer_seq's orientation.
    None if no such length exists (e.g. the local sequence is so AT-rich or
    GC-rich that no length in range hits the target Tm) — callers should
    treat this restriction site as unusable, not silently accept an
    out-of-spec primer.
    """
    tm_lo, tm_hi = tm_range
    extended_candidate = None

    for length in range(min_len, max_len + 1):
        if direction == 1:
            start, end = cut_pos, cut_pos + length
            if end > len(seq):
                break
        else:
            start, end = cut_pos + 1 - length, cut_pos + 1
            if start < 0:
                break
        primer_fwd = seq[start:end]
        primer = primer_fwd if direction == 1 else str(Seq(primer_fwd).reverse_complement())
        tm = simple_tm(primer_fwd)  # orientation-invariant under the Wallace rule
        if tm_lo <= tm <= tm_hi:
            return primer, start, end, tm, False
        if (
            extended_candidate is None
            and tm_hi < tm <= extended_tm_max
            and primer[-1] in "AT"
        ):
            extended_candidate = (primer, start, end, tm, True)

    return extended_candidate


# Common single-cutter enzymes routinely stocked in molecular-biology labs.
# Enzymes earlier in the list are ranked higher when Tm distances are equal.
PREFERRED_ENZYMES: list[str] = [
    "EcoRI", "BamHI", "HindIII", "XhoI", "SalI", "XbaI", "SpeI",
    "NheI",  "KpnI",  "SacI",    "SphI", "PstI", "ClaI", "ApaI",
    "MluI",  "NcoI",  "NdeI",    "BglII","AvrII","MfeI",
    "NotI",  "PacI",  "AscI",    "FseI", "SfiI", "SwaI",
    "AgeI",  "StuI",  "SgrAI",   "BstBI","NarI", "SacII",
]

_PREF_RANK: dict[str, int] = {e: i for i, e in enumerate(PREFERRED_ENZYMES)}
_PREF_DEFAULT = len(PREFERRED_ENZYMES)   # non-preferred enzymes sort last


# NEB catalog info for preferred enzymes.
# cat  = primary catalog number (standard unit; "-HF" suffix = High-Fidelity version available)
# hf   = HF catalog number (None if not available; HF enzymes have less star activity)
# note = any important caveats (methylation sensitivity, etc.)
NEB_CATALOG: dict[str, dict] = {
    "EcoRI":   {"cat": "R0101", "hf": "R3101", "note": ""},
    "BamHI":   {"cat": "R0136", "hf": "R3136", "note": ""},
    "HindIII": {"cat": "R0104", "hf": "R3104", "note": ""},
    "XhoI":    {"cat": "R0146", "hf": None,     "note": ""},
    "SalI":    {"cat": "R0138", "hf": "R3138",  "note": ""},
    "XbaI":    {"cat": "R0145", "hf": None,     "note": "Blocked by Dam methylation (GATC overlap); use dcm-/dam- strain if needed"},
    "SpeI":    {"cat": "R0133", "hf": "R3133",  "note": ""},
    "NheI":    {"cat": "R0131", "hf": "R3131",  "note": ""},
    "KpnI":    {"cat": "R0142", "hf": "R3142",  "note": ""},
    "SacI":    {"cat": "R0156", "hf": "R3156",  "note": ""},
    "SphI":    {"cat": "R0182", "hf": "R3182",  "note": ""},
    "PstI":    {"cat": "R0140", "hf": "R3140",  "note": ""},
    "ClaI":    {"cat": "R0197", "hf": None,     "note": "Blocked by Dam methylation when preceded by G or A (GATCGAT context)"},
    "ApaI":    {"cat": "R0114", "hf": None,     "note": ""},
    "MluI":    {"cat": "R0198", "hf": "R3198",  "note": ""},
    "NcoI":    {"cat": "R0193", "hf": "R3193",  "note": ""},
    "NdeI":    {"cat": "R0111", "hf": "R3111",  "note": ""},
    "BglII":   {"cat": "R0144", "hf": None,     "note": ""},
    "AvrII":   {"cat": "R0174", "hf": None,     "note": ""},
    "MfeI":    {"cat": "R0589", "hf": "R3589",  "note": "Compatible cohesive end with EcoRI"},
    "NotI":    {"cat": "R0189", "hf": "R3189",  "note": "8-bp cutter; rare in most sequences"},
    "PacI":    {"cat": "R0547", "hf": None,     "note": "8-bp cutter; rare in most sequences"},
    "AscI":    {"cat": "R0558", "hf": None,     "note": "8-bp cutter; rare in most sequences"},
    "FseI":    {"cat": "R0588", "hf": None,     "note": "8-bp cutter; rare in most sequences"},
    "SfiI":    {"cat": "R0123", "hf": None,     "note": "Cuts best at 50°C; degenerate site (GGCCNNNNNGGCC)"},
    "SwaI":    {"cat": "R0604", "hf": None,     "note": "8-bp cutter; cuts best at 25°C; rare in most sequences"},
    "AgeI":    {"cat": "R0552", "hf": "R3552",  "note": ""},
    "StuI":    {"cat": "R0187", "hf": None,     "note": "Blunt cutter"},
    "SgrAI":   {"cat": "R0603", "hf": None,     "note": ""},
    "BstBI":   {"cat": "R0519", "hf": None,     "note": "Cuts best at 65°C; thermophilic"},
    "NarI":    {"cat": "R0191", "hf": None,     "note": "Sensitive to CpG methylation"},
    "SacII":   {"cat": "R0157", "hf": None,     "note": "Sensitive to CpG methylation"},
}


def print_neb_table() -> None:
    """Print NEB availability table for all preferred enzymes."""
    print(f"\n{'Enzyme':<10} {'NEB Cat#':<10} {'HF version':<12} Notes")
    print(f"{'-'*10} {'-'*10} {'-'*12} {'-'*50}")
    for enz in PREFERRED_ENZYMES:
        info = NEB_CATALOG.get(enz, {})
        cat  = info.get("cat", "—")
        hf   = info.get("hf") or "—"
        note = info.get("note", "")
        hf_str = hf if hf == "—" else f"{hf} (HF)"
        print(f"  {enz:<10} {cat:<10} {hf_str:<14} {note}")
    print()
    print("All enzymes available at neb.com  |  HF = High-Fidelity (less star activity)")
    print()


def rank_candidates(
    candidates: list[CandidateSite],
    target_tm: float = 51.0,
    preferred: list[str] | None = None,
) -> list[CandidateSite]:
    """
    Sort candidates by a four-key priority:
      1. In-range Tm (48-54°C) before extended-range Tm (54-56°C, A/T 3' end)
      2. |primer_tm - target_tm|  (closest to target wins)
      3. Preference rank (PREFERRED_ENZYMES order; unlisted enzymes last)
      4. Enzyme name alphabetically (deterministic tie-break)

    Pass preferred=[] to disable the preference list.
    """
    pref = _PREF_RANK if preferred is None else {e: i for i, e in enumerate(preferred)}
    pref_default = len(pref)
    return sorted(
        candidates,
        key=lambda c: (c.extended_tm, c.tm_dist, pref.get(c.enzyme, pref_default), c.enzyme),
    )


def design_ad_primers(
    full_seq: str,
    b_start: int,
    c_end: int,
    tm_range: tuple[float, float] = (48.0, 54.0),
    window_bp: tuple[int, int] = (700, 1000),
    min_len: int = 16,
    target_tm: float = 51.0,
    preferred: list[str] | None = None,
) -> tuple[FlankingPrimerResult, FlankingPrimerResult]:
    """
    Design primers A (upstream of B) and D (downstream of C).

    Parameters
    ----------
    full_seq   : complete construct sequence (mutated)
    b_start    : 0-based start of primer B (from Part 4)
    c_end      : 0-based end of primer C template region (from Part 4)
    tm_range   : Wallace Tm target window
    window_bp  : (near, far) distance band from b_start / c_end defining where
                 to search for restriction sites, e.g. (700, 1000) means 700–1000 bp
    min_len    : minimum primer length in bp
    target_tm  : ideal Tm for ranking (°C); default 51 = midpoint of 48-54
    preferred  : ordered list of preferred enzyme names; defaults to PREFERRED_ENZYMES.
                 Pass [] to disable preference ranking.

    Returns
    -------
    (result_a, result_d) — both FlankingPrimerResult
    """
    full_seq = full_seq.upper()
    near, far = window_bp

    # --- Primer A window (upstream of b_start) --------------------------------
    a_win_end   = max(0, b_start - near)
    a_win_start = max(0, b_start - far)

    # --- Primer D window (downstream of c_end) --------------------------------
    d_win_start = min(len(full_seq), c_end + near)
    d_win_end   = min(len(full_seq), c_end + far)

    def _build_candidates(
        window_start: int,
        window_end: int,
        direction: int,
        label: str,
    ) -> FlankingPrimerResult:
        sites = find_unique_sites_in_window(full_seq, window_start, window_end)
        cands: list[CandidateSite] = []
        for enz_name, positions in sites.items():
            for cut_0 in positions:
                walked = design_flanking_primer(
                    full_seq, cut_0, direction, tm_range, min_len
                )
                if walked is None:
                    # No primer length from this site hits Tm 48-54°C (or
                    # the extended A/T-terminus allowance) at >=16nt — not a
                    # usable candidate, skip it rather than ranking an
                    # out-of-spec primer alongside valid ones.
                    continue
                primer_seq, p_start, p_end, p_tm, p_extended = walked
                cands.append(CandidateSite(
                    enzyme=enz_name,
                    cut_pos=cut_0 + 1,   # store 1-based for display
                    cut_pos_0=cut_0,
                    primer_seq=primer_seq,
                    primer_start=p_start,
                    primer_end=p_end,
                    primer_tm=p_tm,
                    tm_dist=abs(p_tm - target_tm),
                    extended_tm=p_extended,
                ))
        ranked = rank_candidates(cands, target_tm, preferred=preferred)
        return FlankingPrimerResult(
            label=label,
            window_start=window_start,
            window_end=window_end,
            candidates=ranked,
        )

    result_a = _build_candidates(a_win_start, a_win_end, direction=+1, label="A")
    result_d = _build_candidates(d_win_start, d_win_end, direction=-1, label="D")

    return result_a, result_d
