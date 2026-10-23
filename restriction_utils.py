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

import re

from Bio.Restriction import RestrictionBatch, Analysis, CommOnly
from Bio.Seq import Seq


# Use NEB-supplied enzymes with 6–8 bp recognition sequences only. Any
# restriction site reported here (in particular a diagnostic site gained or
# lost by a mutation) must be an enzyme a lab can actually order from New
# England Biolabs — Biopython's per-enzyme supplier_list() carries this
# directly from REBASE, so it's checked against real catalog data rather
# than a hand-maintained enzyme list.
def _specified_bases(site: str) -> int:
    """
    Number of fixed (non-N) bases in a recognition site. For an interrupted
    palindrome like BstXI's CCANNNNNNTGG the raw string is 12 characters but
    only 6 bases are actually specified, so that is what determines how
    often the enzyme cuts.
    """
    return sum(1 for c in site.upper() if c != "N")


def _make_batch() -> RestrictionBatch:
    rb = RestrictionBatch(CommOnly)
    # Size filter is about how specifically an enzyme cuts, so it counts
    # SPECIFIED bases, not raw site length. Judging by raw length silently
    # dropped 21 standard NEB enzymes whose sites contain an N spacer —
    # BstXI, SfiI, DraIII, BglI, AlwNI, XmnI and friends. They are common
    # in the lab and often give the cheapest diagnostic (no extra silent
    # mutation needed), so excluding them forced longer, costlier designs.
    return RestrictionBatch(
        [e for e in rb
         if (6 <= len(e.site) <= 8 or 6 <= _specified_bases(e.site) <= 8)
         and "New England Biolabs" in e.supplier_list()]
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


def predict_gel_bands(
    seq: str, enzyme_name: str, circular: bool = True,
) -> list[int]:
    """
    Predicted fragment sizes (bp) from digesting seq with one named enzyme
    — what would actually run on a gel. Most cloning plasmids are
    maintained and digested as circular molecules, so this defaults to
    circular topology: a site straddling the sequence's start/end (the
    origin, as stored in this linear FASTA representation) is still found,
    and the fragment "wrapping around" the origin is computed correctly.

    Band count follows real digest topology:
      0 cut sites  -> one band, the whole (uncut) length
      1 cut site   -> one band, the whole length (a single cut just
                      linearizes a circular molecule — no size change)
      N>=2 sites   -> N bands, from the gaps between consecutive cut
                      positions (including the wrap-around gap for
                      circular topology)
    For circular=False (a genuinely linear construct), N cut sites always
    give N+1 fragments instead, with no wrap-around gap.

    Returns band sizes sorted largest-first (how they'd run on a gel, top
    to bottom), or [] if the enzyme name isn't recognized.
    """
    enz = _ENZYME_BY_NAME.get(enzyme_name)
    if enz is None:
        return []
    analysis = Analysis(RestrictionBatch([enz]), Seq(seq), linear=not circular)
    raw = analysis.full()
    cuts = sorted(raw.get(enz, []))
    n = len(seq)

    if not cuts:
        return [n]
    if circular and len(cuts) == 1:
        return [n]

    bands = [b - a for a, b in zip(cuts, cuts[1:])]
    if circular:
        bands.append(n - cuts[-1] + cuts[0])
    else:
        bands.insert(0, cuts[0])
        bands.append(n - cuts[-1])
    return sorted(bands, reverse=True)


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


_METHYL_MOTIFS = {
    "Dam (GATC)": re.compile(r"(?=GATC)"),
    "Dcm (CCWGG)": re.compile(r"(?=CC[AT]GG)"),
}
_METHYL_MOTIF_LEN = {"Dam (GATC)": 4, "Dcm (CCWGG)": 5}


def methylation_overlaps(seq: str, enzyme_name: str, near_start: int, near_end: int) -> list[str]:
    """
    Which E. coli methylation motifs (Dam GATC, Dcm CCWGG) overlap this
    enzyme's recognition site(s) lying within [near_start, near_end) of seq.

    This only reports the sequence context; whether an overlap actually
    blocks cutting depends on the enzyme's own Dam/Dcm sensitivity (see
    NEB's methylation sensitivity table). An empty list means no overlap,
    so methylation cannot interfere at this site.
    """
    enz = _ENZYME_BY_NAME.get(enzyme_name)
    if enz is None:
        return []
    seq = seq.upper()
    sites = [
        (m.start(), m.start() + enz.size)
        for m in enz.compsite.finditer(seq)
        if m.start() < near_end and m.start() + enz.size > near_start
    ]
    hits = []
    for label, motif in _METHYL_MOTIFS.items():
        mlen = _METHYL_MOTIF_LEN[label]
        lo = max(0, min((s for s, _ in sites), default=0) - mlen)
        hi = max((e for _, e in sites), default=0) + mlen
        for m in motif.finditer(seq, lo, hi):
            if any(m.start() < e and m.start() + mlen > s for s, e in sites):
                hits.append(label)
                break
    return hits


# Gel-readability thresholds. A diagnostic digest does not need an enzyme
# that cuts nowhere else: an enzyme with existing sites is perfectly usable
# as long as the BAND PATTERN visibly changes (e.g. a 2482 bp band splitting
# into 1861 + 621 against unchanged 4941 and 482 bands). Insisting on a
# globally unique site instead forces extra silent mutations and longer
# primers, which is worse.
_MIN_VISIBLE_BAND = 150     # bp; smaller fragments run off / are too faint
_MIN_BAND_SEPARATION = 0.10  # two bands within 10% co-migrate as one
_MAX_READABLE_BANDS = 12     # beyond this a digest is a smear


def _comigrates(band: int, lane: list[int]) -> bool:
    """True if another band in the same lane runs too close to this one."""
    near = sum(1 for other in lane
               if abs(other - band) <= _MIN_BAND_SEPARATION * max(other, band))
    return near > 1   # itself, plus at least one other


def gel_pattern_change(
    wt_seq: str,
    mut_seq: str,
    enzyme_name: str,
    circular: bool = True,
) -> dict:
    """
    Compare the predicted digest of wild type vs mutant for one enzyme and
    judge whether the difference is actually readable on a gel.

    Returns a dict with:
      wt_bands, mut_bands : predicted fragment sizes (bp, largest first)
      lost_bands          : bands present in WT but not the mutant
      gained_bands        : bands present in the mutant but not WT
      readable            : True if the change would be unambiguous on a gel
      reason              : short explanation (why not, when readable is False)

    "Readable" requires every band that appears or disappears to be big
    enough to see, to not co-migrate with another band in its own lane, and
    the digest not to have so many fragments that it smears.
    """
    from collections import Counter

    wt = predict_gel_bands(wt_seq, enzyme_name, circular)
    mut = predict_gel_bands(mut_seq, enzyme_name, circular)
    info = {"wt_bands": wt, "mut_bands": mut, "lost_bands": [],
            "gained_bands": [], "readable": False, "reason": ""}
    if not wt or not mut:
        info["reason"] = "enzyme not recognized"
        return info

    cw, cm = Counter(wt), Counter(mut)
    lost = sorted((cw - cm).elements(), reverse=True)
    gained = sorted((cm - cw).elements(), reverse=True)
    info["lost_bands"], info["gained_bands"] = lost, gained

    if not lost and not gained:
        info["reason"] = "digest pattern is identical — nothing to see"
        return info
    if max(len(wt), len(mut)) > _MAX_READABLE_BANDS:
        info["reason"] = f"too many fragments ({max(len(wt), len(mut))}) to read"
        return info
    small = [b for b in lost + gained if b < _MIN_VISIBLE_BAND]
    if small:
        info["reason"] = f"changed fragment {min(small)} bp is too small to see"
        return info
    for band in gained:
        if _comigrates(band, mut):
            info["reason"] = f"new {band} bp band co-migrates with another band"
            return info
    for band in lost:
        if _comigrates(band, wt):
            info["reason"] = f"the {band} bp band that disappears co-migrates with another"
            return info

    info["readable"] = True
    return info
