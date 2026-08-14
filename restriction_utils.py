"""
Part 3 — Restriction site scanning and silent-mutation search.

scan_sites(seq)
  -> dict[enzyme_name, list[int]]   (1-based cut positions)

gained_lost_sites(seq_before, seq_after)
  -> dict with keys 'gained' and 'lost', each a dict[name -> positions]

find_silent_restriction_sites(seq, orf_start, window_start, window_end, new_aa=None)
  -> list of dicts describing silent mutations that create/destroy a site,
     sorted by number of nucleotide changes (ascending)
"""

from Bio.Restriction import RestrictionBatch, Analysis, CommOnly
from Bio.Seq import Seq


# Use commercially-available enzymes with 6–8 bp recognition sequences
def _make_batch() -> RestrictionBatch:
    rb = RestrictionBatch(CommOnly)
    return RestrictionBatch(
        [e for e in rb if 6 <= len(e.site) <= 8]
    )


_BATCH = _make_batch()


def _compute_max_flank(batch) -> int:
    """
    Largest distance (nt) from the start of any enzyme's recognition site to
    its reported cut position, across the whole batch. Some Type IIS/IIG
    enzymes (e.g. MmeI, NmeAIII) cut tens of bases away from their
    recognition sequence — any local-window scan must pad by at least this
    much on both sides, or such enzymes' sites will silently go undetected
    near the window edges.
    """
    max_flank = 0
    for e in batch:
        offset = max(abs(e.fst5), abs(e.fst3))
        max_flank = max(max_flank, len(e.site) + offset)
    return max_flank


# Computed once from the actual enzyme batch rather than assumed from
# recognition-site length alone (see _compute_max_flank docstring).
_MAX_ENZYME_FLANK = _compute_max_flank(_BATCH)


def scan_sites(seq: str) -> dict[str, list[int]]:
    """
    Return all cut positions for all 6-8bp commercial restriction enzymes.
    Positions are 1-based (Biopython default).
    Only enzymes with at least one cut are returned.
    """
    analysis = Analysis(_BATCH, Seq(seq), linear=True)
    raw = analysis.full()
    return {str(enz): positions for enz, positions in raw.items() if positions}


_ENZYME_BY_NAME = {str(e): e for e in _BATCH}


def count_sites(seq: str, enzyme_name: str) -> list[int]:
    """
    Cut positions for a SINGLE named enzyme only (1-based). Much cheaper
    than scan_sites() when only one enzyme's count is actually needed —
    scan_sites analyzes the whole ~50-80 enzyme batch every call, which
    gets expensive when called repeatedly (e.g. once per candidate while
    searching for a globally-unique diagnostic site).
    """
    enz = _ENZYME_BY_NAME.get(enzyme_name)
    if enz is None:
        return []
    analysis = Analysis(RestrictionBatch([enz]), Seq(seq), linear=True)
    raw = analysis.full()
    return raw.get(enz, [])


def gained_lost_sites(
    seq_before: str,
    seq_after: str,
) -> dict[str, dict[str, list[int]]]:
    """
    Compare restriction maps of two sequences.
    Returns {'gained': {name: [pos, ...]}, 'lost': {name: [pos, ...]}}
    """
    before = scan_sites(seq_before)
    after = scan_sites(seq_after)

    all_enzymes = set(before) | set(after)
    gained: dict[str, list[int]] = {}
    lost: dict[str, list[int]] = {}

    for name in all_enzymes:
        b_sites = set(before.get(name, []))
        a_sites = set(after.get(name, []))
        new = sorted(a_sites - b_sites)
        gone = sorted(b_sites - a_sites)
        if new:
            gained[name] = new
        if gone:
            lost[name] = gone

    return {"gained": gained, "lost": lost}


# ---------------------------------------------------------------------------
# Silent mutation search
# ---------------------------------------------------------------------------

from Bio.Data import CodonTable as _CT
from itertools import product as _product

_FWD = _CT.standard_dna_table.forward_table   # codon -> aa
_STOP = set(_CT.standard_dna_table.stop_codons)


def _synonymous_codons(codon: str) -> list[str]:
    """All codons (including original) that encode the same amino acid."""
    codon = codon.upper()
    if codon in _STOP:
        return [codon]
    aa = _FWD.get(codon)
    if aa is None:
        return [codon]
    return [c for c, a in _FWD.items() if a == aa]


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def find_silent_restriction_sites(
    seq: str,
    orf_start: int,
    window_start: int,
    window_end: int,
) -> list[dict]:
    """
    Search for single-codon silent mutations within [window_start, window_end)
    (0-based, absolute positions in seq) that create or destroy a restriction site.

    Each result dict contains:
      position      : 0-based absolute start of the altered codon
      aa_index      : 1-based amino acid index within the ORF
      original_codon: str
      new_codon     : str
      changes       : int (number of nt differences)
      effect        : 'gained' or 'lost'
      enzyme        : enzyme name
      site_position : 1-based position of the restriction site in mutated seq

    Results are sorted by changes (ascending), then enzyme name.
    """
    seq = seq.upper()
    results = []

    # A codon swap (3 nt) can only create/destroy a restriction site that
    # overlaps those 3 nt. Scanning a small local window instead of the
    # whole sequence avoids a full Bio.Restriction.Analysis() pass
    # (O(sequence length) per enzyme) for every single candidate codon —
    # this was previously the main cost and made widened searches on long
    # sequences extremely slow. PAD must cover the largest distance from a
    # recognition site to its reported cut position across the whole
    # enzyme batch (some Type IIS/IIG enzymes like MmeI cut tens of bases
    # away from their site) — see _compute_max_flank.
    PAD = _MAX_ENZYME_FLANK

    # Identify codons that overlap the window
    # orf_start is the first base of the first codon
    # First codon index (0-based aa) whose codon overlaps [window_start, window_end)
    first_aa_0 = max(0, (window_start - orf_start) // 3)
    last_aa_0 = (window_end - orf_start - 1) // 3

    for aa_0 in range(first_aa_0, last_aa_0 + 1):
        codon_abs = orf_start + aa_0 * 3
        if codon_abs + 3 > len(seq):
            break
        original_codon = seq[codon_abs: codon_abs + 3]
        synonyms = _synonymous_codons(original_codon)

        local_start = max(0, codon_abs - PAD)
        local_end   = min(len(seq), codon_abs + 3 + PAD)
        local_offset_in_codon = codon_abs - local_start
        local_before = seq[local_start:local_end]
        baseline_local = scan_sites(local_before)

        for alt_codon in synonyms:
            if alt_codon == original_codon:
                continue
            local_after = (
                local_before[:local_offset_in_codon]
                + alt_codon
                + local_before[local_offset_in_codon + 3:]
            )
            after_local = scan_sites(local_after)

            all_enzymes = set(baseline_local) | set(after_local)
            for enz_name in all_enzymes:
                b_sites = set(baseline_local.get(enz_name, []))
                a_sites = set(after_local.get(enz_name, []))
                gained_pos = sorted(p + local_start for p in (a_sites - b_sites))
                lost_pos   = sorted(p + local_start for p in (b_sites - a_sites))
                if gained_pos:
                    results.append({
                        "position": codon_abs,
                        "aa_index": aa_0 + 1,
                        "original_codon": original_codon,
                        "new_codon": alt_codon,
                        "changes": _hamming(original_codon, alt_codon),
                        "effect": "gained",
                        "enzyme": enz_name,
                        "site_positions": gained_pos,
                    })
                if lost_pos:
                    results.append({
                        "position": codon_abs,
                        "aa_index": aa_0 + 1,
                        "original_codon": original_codon,
                        "new_codon": alt_codon,
                        "changes": _hamming(original_codon, alt_codon),
                        "effect": "lost",
                        "enzyme": enz_name,
                        "site_positions": lost_pos,
                    })

    results.sort(key=lambda r: (r["changes"], r["enzyme"]))
    return results
