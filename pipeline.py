"""
Part 7a — Unified pipeline function.

design_mutation_primers(sequence, orf_start, target_position, original_aa, new_aa, ...)
    Runs Parts 1-6 in order and returns a PipelineResult dataclass.

Failure modes are collected into PipelineResult.errors rather than raised,
so the caller (CLI or notebook) can decide how to present them.

Recoverable warnings (e.g. no diagnostic site, no flanking sites for one
primer) are stored in PipelineResult.warnings and the pipeline continues
with graceful degradation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from Bio.Seq import Seq

from codon_utils import find_codon_and_mutate, ranked_codon_options, apply_codon
from primer_bc import design_bc_primers, PrimerBCResult
from primer_ad import design_ad_primers, FlankingPrimerResult, CandidateSite, PREFERRED_ENZYMES
from tm_utils import simple_tm
from restriction_utils import gained_lost_sites, find_silent_restriction_sites, scan_sites, count_sites
from assembly import (
    assemble_product,
    translate_orf,
    run_full_verification,
    VerificationReport,
)


# ---------------------------------------------------------------------------
# Structured result
# ---------------------------------------------------------------------------

@dataclass
class PrimerInfo:
    sequence: str
    tm: float                        # Tm of the full primer (Wallace)
    length: int
    start: int                       # 0-based, in the construct
    end: int                         # 0-based exclusive
    enzyme: str | None = None        # flanking enzyme, if applicable
    cut_pos: int | None = None       # 1-based cut position in construct
    tm_anneal: float | None = None   # annealing-region Tm (primer C only)


@dataclass
class DiagnosticInfo:
    enzyme: str
    effect: str                      # 'gained' or 'lost'
    source: str                      # 'mutation' or 'silent_mutation'
    # Set when source == 'silent_mutation'
    silent_aa_index: int | None = None
    silent_original_codon: str | None = None
    silent_new_codon: str | None = None
    silent_changes: int | None = None


@dataclass
class PipelineResult:
    # ── Input echo ────────────────────────────────────────────────────────
    mutation_label: str          # e.g. "K15E", or "K15E + R255A" for a combined mutation
    orf_start: int
    original_codon: str          # " / "-joined for multiple mutations
    new_codon: str                # " / "-joined for multiple mutations
    changed_positions: list[int]
    orf_start_detected: int = 0       # the orf_start actually used (auto or supplied)
    mutation_nt_position: int = 0     # 0-based nucleotide position of first changed base

    # ── Structured per-mutation detail (length 1 for a single mutation) ────
    mutation_positions: list[int] = field(default_factory=list)
    original_aas: list[str] = field(default_factory=list)
    new_aas: list[str] = field(default_factory=list)
    original_codons: list[str] = field(default_factory=list)
    new_codons: list[str] = field(default_factory=list)

    # ── Primers ───────────────────────────────────────────────────────────
    primer_A: PrimerInfo | None = None
    primer_B: PrimerInfo | None = None
    primer_C: PrimerInfo | None = None
    primer_D: PrimerInfo | None = None
    overlap_seq: str = ""
    overlap_tm: float = 0.0

    # ── Diagnostic restriction site ───────────────────────────────────────
    diagnostic: DiagnosticInfo | None = None

    # ── Full cutting-pattern diff: every NEB enzyme gained/lost anywhere in
    #    the construct by the mutation (not just the one picked as the
    #    diagnostic) — {"gained": {enzyme: [1-based positions in mutated
    #    seq]}, "lost": {enzyme: [1-based positions in wild-type seq]}}
    cut_site_diff: dict = field(default_factory=lambda: {"gained": {}, "lost": {}})

    # ── Fragment sizes ────────────────────────────────────────────────────
    frag_ab_bp: int = 0
    frag_cd_bp: int = 0
    frag_ad_bp: int = 0    # total assembled product

    # ── Fragment sequences (5'→3', sense strand) ────────────────────────────
    frag_ab_seq: str = ""
    frag_cd_seq: str = ""
    frag_ad_seq: str = ""  # the full assembled PCR product

    # ── Full mutated construct (5'→3', includes any silent mutation too) ───
    mutated_sequence: str = ""
    original_sequence: str = ""  # the input construct, unmodified (wild type)

    # ── Verification ──────────────────────────────────────────────────────
    translation_passed: bool | None = None
    restriction_passed: bool | None = None
    overall_passed: bool | None = None
    translation_detail: str = ""
    restriction_detail: str = ""

    # ── Internals (kept for downstream use / debugging) ───────────────────
    report: VerificationReport | None = None
    bc_result: PrimerBCResult | None = None
    ad_result_a: FlankingPrimerResult | None = None
    ad_result_d: FlankingPrimerResult | None = None

    # ── Status ────────────────────────────────────────────────────────────
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no fatal errors occurred and all verifications passed."""
        return (
            len(self.errors) == 0
            and self.overall_passed is True
        )


# ---------------------------------------------------------------------------
# Mutation-label parser
# ---------------------------------------------------------------------------

_MUTATION_RE = re.compile(
    r"^([A-Z\*])(\d+)([A-Z\*])$",
    re.IGNORECASE,
)


def parse_mutation_label(label: str) -> tuple[str, int, str]:
    """
    Parse a mutation label such as 'K255E' or 'k255e'.
    Returns (original_aa, position_1based, new_aa) as upper-case strings.
    Raises ValueError on bad format.
    """
    m = _MUTATION_RE.match(label.strip())
    if not m:
        raise ValueError(
            f"Cannot parse mutation '{label}'. "
            "Expected format: <original_AA><position><new_AA>, e.g. K255E"
        )
    return m.group(1).upper(), int(m.group(2)), m.group(3).upper()


def parse_mutation_labels(label: str) -> list[tuple[str, int, str]]:
    """
    Parse one or more '+'-separated mutation labels, e.g. 'K255E' or
    'K255E+R300A' for a combined double mutation.
    Returns a list of (original_aa, position_1based, new_aa) tuples.
    Raises ValueError on bad format, or on duplicate positions.
    """
    parts = [p.strip() for p in label.split("+") if p.strip()]
    if not parts:
        raise ValueError(f"Cannot parse mutation '{label}'.")
    mutations = [parse_mutation_label(p) for p in parts]
    positions = [pos for _, pos, _ in mutations]
    if len(set(positions)) != len(positions):
        raise ValueError(
            f"Duplicate amino acid position(s) in '{label}' — each mutation "
            "must target a different position."
        )
    return mutations


# ---------------------------------------------------------------------------
# ORF auto-detection and position finding
# ---------------------------------------------------------------------------

def find_all_positions(
    sequence: str,
    orf_start: int,
    original_aa: str,
) -> list[tuple[int, str]]:
    """
    Translate the ORF and return every position where original_aa occurs.
    Returns list of (1-based position, codon) tuples.
    """
    sequence = sequence.upper()
    original_aa = original_aa.upper()
    orf = sequence[orf_start:]
    results = []
    for i in range(len(orf) // 3):
        codon = orf[i*3 : i*3+3]
        if str(Seq(codon).translate()) == original_aa:
            results.append((i + 1, codon))
        if codon in ("TAA", "TAG", "TGA"):
            break
    return results


def find_orf_start(
    sequence: str,
    original_aa: str,
    target_position: int,
) -> list[int]:
    """
    Scan every ATG in sequence and return the 0-based positions of those where
    translating from that ATG places original_aa at target_position (1-based).

    Returns a list of candidate orf_start positions (may be empty or have
    multiple hits if the sequence is ambiguous).
    """
    sequence = sequence.upper()
    original_aa = original_aa.upper()
    aa_idx = target_position - 1          # 0-based codon index
    candidates = []

    pos = 0
    while True:
        atg = sequence.find("ATG", pos)
        if atg == -1:
            break
        codon_start = atg + aa_idx * 3
        if codon_start + 3 <= len(sequence):
            codon = sequence[codon_start: codon_start + 3]
            if str(Seq(codon).translate()) == original_aa:
                candidates.append(atg)
        pos = atg + 1

    return candidates


# ---------------------------------------------------------------------------
# Graceful error helpers
# ---------------------------------------------------------------------------

class _PipelineError(Exception):
    """Raised internally to abort the pipeline on a fatal error."""


def _locate_user_primer(working_seq: str, primer_seq: str, label: str) -> tuple[int, int, str]:
    """
    Find a user-supplied primer A/D sequence within working_seq.

    Tries the sequence as given first, then its reverse complement, since a
    user might paste either orientation. Either match is equally valid for
    locating the span — Tm is orientation-invariant (a sequence and its
    reverse complement have identical G/C and A/T counts under the Wallace
    rule) — but the oligo actually stored/displayed is always normalized to
    the orientation that primer needs: primer A is the forward primer, so
    its oligo is the template-strand span unmodified; primer D closes
    fragment "cd" back toward C and so is the reverse primer — its real
    oligo is the reverse complement of the template-strand span, regardless
    of which orientation the user happened to paste.

    Returns (start, end, seq_as_stored) — start/end describe the template-
    strand span (needed for fragment position math); seq_as_stored is the
    correctly-oriented oligo for that primer's role.
    """
    primer_seq = primer_seq.upper().strip()
    if not primer_seq:
        raise _PipelineError(f"Primer {label}: an empty sequence was supplied.")

    revcomp = str(Seq(primer_seq).reverse_complement())
    fwd_count = working_seq.count(primer_seq)
    rev_count = working_seq.count(revcomp) if revcomp != primer_seq else 0

    if fwd_count + rev_count == 0:
        raise _PipelineError(
            f"Primer {label} sequence not found in the construct — checked "
            f"both as given and as its reverse complement. Double-check it "
            f"was copied correctly and matches this sequence."
        )
    if fwd_count + rev_count > 1:
        raise _PipelineError(
            f"Primer {label} sequence is ambiguous — it (or its reverse "
            f"complement) appears {fwd_count + rev_count} times in the "
            f"construct. Supply a longer, unique primer sequence."
        )

    if fwd_count == 1:
        start = working_seq.index(primer_seq)
        end = start + len(primer_seq)
    else:
        start = working_seq.index(revcomp)
        end = start + len(revcomp)

    template_span = working_seq[start:end]
    seq_as_stored = template_span if label == "A" else str(Seq(template_span).reverse_complement())
    return start, end, seq_as_stored


def _silent_mutation_seq(
    seq: str,
    codon_abs: int,
    new_codon: str,
) -> str:
    seq = seq.upper()
    return seq[:codon_abs] + new_codon.upper() + seq[codon_abs + 3:]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def design_mutation_primers(
    sequence: str,
    target_position: int | list[int],
    original_aa: str | list[str],
    new_aa: str | list[str],
    *,
    orf_start: int | None = None,
    tm_range: tuple[float, float] = (48.0, 54.0),
    window_bp: tuple[int, int] = (350, 500),
    min_overlap: int = 16,
    target_tm: float = 51.0,
    preferred_enzymes: list[str] | None = None,
    silent_window_flank: int = 9,
    max_silent_search_flank: int = 150,
    max_ad_window_expansions: int = 3,
    primer_A_seq: str | None = None,
    primer_D_seq: str | None = None,
) -> PipelineResult:
    """
    Full Gibson primer design pipeline (Parts 1–6).

    Parameters
    ----------
    sequence         : DNA sequence of the full construct (any case; IUPAC A/C/G/T)
    target_position  : 1-based amino acid number to mutate. Pass a list of
                       positions (with matching lists for original_aa and
                       new_aa) to design a combined multi-mutation — all
                       mutations must fall close enough together to share
                       one B/C overlap region (a double mutation a few
                       codons apart is realistic; one spanning hundreds of
                       bp is not — the overlap Tm/length search will simply
                       fail to find a window, same as it would by hand).
    original_aa      : single-letter AA expected at that position (confirms identity)
    new_aa           : desired amino acid
    orf_start        : 0-based position where the ORF begins (the A in ATG).
                       If None, the sequence is scanned automatically and the
                       unique ATG that places original_aa at target_position is
                       used. Auto-detection only supports a single mutation —
                       orf_start must be supplied explicitly for a multi-mutation
                       design.
    tm_range         : Wallace Tm window for all primer walks (default 48–54°C)
    window_bp        : (near, far) bp distance from the B/C region to search for
                       a flanking restriction site, applied on EACH side
                       independently (upstream for A, downstream for D).
                       Default (350, 500) targets the protocol's "~700-1000bp
                       apart" between sites A and D in total (near+near to
                       far+far), not 700-1000bp on each individual side.
    min_overlap      : minimum B/C overlap length in bp
    target_tm        : ideal primer Tm for ranking A/D candidates (default 51°C)
    preferred_enzymes: ordered preference list for A/D enzyme selection;
                       None → use the built-in PREFERRED_ENZYMES list
    silent_window_flank: starting nt to scan on each side of the mutation codon
                         when looking for a silent diagnostic restriction site.
                         If nothing is found, the window widens automatically
                         (×3 each retry) up to max_silent_search_flank.
    max_silent_search_flank: cap (nt) on how far the diagnostic-site search
                         will widen before giving up.
    max_ad_window_expansions: how many times to push the primer A/D search
                         window further out (default window_bp shifted +50%
                         of its span each time) if no unique flanking site
                         is found. Does NOT introduce any mutations — only
                         widens where it looks.
    primer_A_seq/primer_D_seq: supply your own primer A and/or D sequence
                         instead of having the tool search for a restriction
                         site. Each is located in the construct (as given, or
                         as its reverse complement) and used directly — the
                         restriction-site search for that side is skipped
                         entirely. Either or both may be supplied; any side
                         left as None is still auto-designed as usual.

    Returns
    -------
    PipelineResult — always returned, even on error.
    Check result.errors for fatal failures; result.warnings for soft ones.
    """
    sequence = sequence.upper().strip()

    positions      = list(target_position) if isinstance(target_position, (list, tuple)) else [target_position]
    original_aas   = [a.upper() for a in original_aa] if isinstance(original_aa, (list, tuple)) else [original_aa.upper()]
    new_aas        = [a.upper() for a in new_aa]      if isinstance(new_aa, (list, tuple))      else [new_aa.upper()]
    if not (len(positions) == len(original_aas) == len(new_aas)):
        raise ValueError(
            f"target_position, original_aa, and new_aa must all be the same "
            f"length ({len(positions)}, {len(original_aas)}, {len(new_aas)})."
        )
    is_multi = len(positions) > 1
    mutation_label = " + ".join(
        f"{oa}{p}{na}" for p, oa, na in zip(positions, original_aas, new_aas)
    )

    result = PipelineResult(
        mutation_label=mutation_label,
        orf_start=orf_start or 0,
        original_codon="",
        new_codon="",
        changed_positions=[],
        original_sequence=sequence,
    )

    try:
        # ── Step 0: resolve orf_start and validate inputs ────────────────────
        for pos in positions:
            if pos < 1:
                raise _PipelineError("target_position must be ≥ 1.")

        # Protocol's stated minimum B/C overlap length is a hard floor, not a
        # suggestion — enforced here (not just documented) so no caller,
        # including a user-supplied --min-overlap below 16, can silently
        # produce an out-of-spec overlap.
        if min_overlap < 16:
            result.warnings.append(
                f"min_overlap={min_overlap} is below the protocol's minimum "
                f"of 16bp — raised to 16."
            )
            min_overlap = 16

        if orf_start is None:
            if is_multi:
                raise _PipelineError(
                    "orf_start must be supplied explicitly for a multi-mutation "
                    "design (auto-detection only supports a single mutation)."
                )
            # Auto-detect: find every ATG where position target_position is original_aa
            candidates = find_orf_start(sequence, original_aas[0], positions[0])
            if len(candidates) == 0:
                raise _PipelineError(
                    f"Could not find an ATG in the sequence where amino acid "
                    f"{positions[0]} is {original_aas[0]}. "
                    "Check that the mutation label matches the sequence, "
                    "or supply --orf-start manually."
                )
            if len(candidates) > 1:
                positions_str = ", ".join(str(c) for c in candidates)
                raise _PipelineError(
                    f"Found {len(candidates)} ATGs where position {positions[0]} "
                    f"is {original_aas[0]} (at nt positions {positions_str}). "
                    "Sequence is ambiguous — supply --orf-start to specify which one."
                )
            orf_start = candidates[0]
            result.orf_start = orf_start

        if orf_start < 0 or orf_start >= len(sequence):
            raise _PipelineError(
                f"orf_start {orf_start} is outside sequence length {len(sequence)}."
            )

        codon_abs_list = []
        for pos, oaa in zip(positions, original_aas):
            codon_abs = orf_start + (pos - 1) * 3
            if codon_abs + 3 > len(sequence):
                raise _PipelineError(
                    f"Codon for position {pos} extends past the end "
                    f"of the sequence (need pos {codon_abs}–{codon_abs+2}, "
                    f"sequence is {len(sequence)} bp)."
                )
            actual_codon = sequence[codon_abs: codon_abs + 3]
            actual_aa = str(Seq(actual_codon).translate())
            if actual_aa != oaa:
                raise _PipelineError(
                    f"Position {pos}: expected {oaa} "
                    f"but found {actual_aa} (codon {actual_codon}). "
                    "Check orf_start and target_position."
                )
            codon_abs_list.append(codon_abs)

        result.orf_start_detected = orf_start

        # ── Part 1: codon mutation (applied sequentially for each position;
        #    substitutions preserve length, so earlier positions' absolute
        #    offsets remain valid for later ones) ─────────────────────────────
        mutated_seq = sequence
        orig_codons: list[str] = []
        new_codons_list: list[str] = []
        changed_pos: list[int] = []
        for pos, naa in zip(positions, new_aas):
            try:
                mutated_seq, cp, oc, nc = find_codon_and_mutate(
                    mutated_seq, orf_start, pos, naa
                )
            except ValueError as exc:
                raise _PipelineError(f"Codon mutation failed: {exc}") from exc
            orig_codons.append(oc)
            new_codons_list.append(nc)
            changed_pos.extend(cp)
        changed_pos = sorted(changed_pos)

        # Single-mutation-only convenience aliases used further down for the
        # "try an alternate codon" diagnostic optimization.
        orig_codon = orig_codons[0]
        new_codon = new_codons_list[0]
        codon_abs = codon_abs_list[0]

        result.original_codon = " / ".join(orig_codons)
        result.new_codon = " / ".join(new_codons_list)
        result.original_codons = orig_codons
        result.new_codons = new_codons_list
        result.mutation_positions = positions
        result.original_aas = original_aas
        result.new_aas = new_aas
        result.changed_positions = changed_pos
        result.mutation_nt_position = changed_pos[0] if changed_pos else codon_abs_list[0]

        # ── Part 3: find diagnostic restriction site ──────────────────────────
        diff = gained_lost_sites(sequence, mutated_seq)
        gained = diff["gained"]
        lost   = diff["lost"]

        diagnostic: DiagnosticInfo | None = None
        working_seq = mutated_seq   # may be updated by silent mutation below

        # A "gained"/"lost" enzyme is only a clean, useful diagnostic if it
        # doesn't ALSO cut somewhere else unrelated to the mutation — otherwise
        # a digest cuts either way and the presence/absence check verified in
        # Part 6 is meaningless (e.g. an enzyme with 4 sites total, one of
        # which happens to be destroyed by the mutation, still visibly cuts
        # the construct 3 other times regardless of the mutation).
        before_sites = scan_sites(sequence)
        after_sites  = scan_sites(mutated_seq)
        clean_gained = [
            enz for enz in gained
            if len(before_sites.get(enz, [])) == 0
            and len(after_sites.get(enz, [])) == len(gained[enz])
        ]
        clean_lost = [
            enz for enz in lost
            if len(after_sites.get(enz, [])) == 0
            and len(before_sites.get(enz, [])) == len(lost[enz])
        ]

        if clean_gained:
            enz = clean_gained[0]
            diagnostic = DiagnosticInfo(
                enzyme=enz, effect="gained", source="mutation"
            )
        elif clean_lost:
            enz = clean_lost[0]
            diagnostic = DiagnosticInfo(
                enzyme=enz, effect="lost", source="mutation"
            )
        elif not is_multi:
            # Protocol step 1a: "change the nucleotide again" — before
            # resorting to an adjacent-codon silent mutation, try OTHER
            # codons that still encode the same target amino acid (there is
            # often more than one, and the fewest-change pick above isn't
            # necessarily the one that creates/destroys a usable site).
            # Only attempted for a single mutation — combinatorially trying
            # alternate codons at every position of a multi-mutation design
            # isn't worth the complexity; multi-mutation designs fall
            # straight through to the adjacent-codon silent search below.
            for alt_codon in ranked_codon_options(orig_codon, new_aas[0]):
                alt_seq, alt_changed_pos, _, _ = apply_codon(
                    sequence, orf_start, positions[0], alt_codon
                )
                alt_diff = gained_lost_sites(sequence, alt_seq)
                alt_after = scan_sites(alt_seq)
                alt_clean_gained = [
                    enz for enz in alt_diff["gained"]
                    if len(before_sites.get(enz, [])) == 0
                    and len(alt_after.get(enz, [])) == len(alt_diff["gained"][enz])
                ]
                alt_clean_lost = [
                    enz for enz in alt_diff["lost"]
                    if len(alt_after.get(enz, [])) == 0
                    and len(before_sites.get(enz, [])) == len(alt_diff["lost"][enz])
                ]
                if alt_clean_gained or alt_clean_lost:
                    mutated_seq, changed_pos, new_codon = alt_seq, alt_changed_pos, alt_codon
                    working_seq = mutated_seq
                    after_sites = alt_after
                    result.new_codon = new_codon
                    result.new_codons = [new_codon]
                    result.changed_positions = changed_pos
                    result.mutation_nt_position = changed_pos[0] if changed_pos else codon_abs
                    if alt_clean_gained:
                        diagnostic = DiagnosticInfo(
                            enzyme=alt_clean_gained[0], effect="gained", source="mutation"
                        )
                    else:
                        diagnostic = DiagnosticInfo(
                            enzyme=alt_clean_lost[0], effect="lost", source="mutation"
                        )
                    result.warnings.append(
                        f"Used alternate codon {new_codon} for {new_aas[0]} (instead of "
                        f"the fewest-change option) to obtain a clean diagnostic "
                        f"site ({diagnostic.enzyme})."
                    )
                    break

        if diagnostic is None:
            # Search for a silent mutation, widening the window outward from
            # the mutation codon until a hit is found or the cap is reached.
            # Capped (default 150 nt each side ≈ 50 codons) so the diagnostic
            # site stays close enough to the mutation to be a useful screen.
            silent_hits: list = []
            flank = silent_window_flank
            searched_flank = flank
            span_start = min(codon_abs_list)
            span_end   = max(c + 3 for c in codon_abs_list)

            def _clean_candidate(candidate: dict) -> bool:
                # Same uniqueness concern as the direct-mutation diagnostic
                # above: a candidate enzyme found via the local-window search
                # could still have other pre-existing sites elsewhere in the
                # full construct, making it a useless gained/lost signal on a
                # real digest. Uses count_sites (single enzyme) rather than
                # scan_sites (all ~50-80 enzymes) — this runs once per
                # candidate at every widen step, so scanning the whole batch
                # each time made widened searches extremely slow.
                enz = candidate["enzyme"]
                cand_seq = _silent_mutation_seq(
                    mutated_seq, candidate["position"], candidate["new_codon"]
                )
                cand_count = len(count_sites(cand_seq, enz))
                enz_before = len(after_sites.get(enz, []))
                if candidate["effect"] == "gained":
                    return enz_before == 0 and cand_count == len(candidate["site_positions"])
                else:  # "lost"
                    return cand_count == 0 and enz_before == len(candidate["site_positions"])

            clean_hit = None
            best_unclean_hit = None  # best-effort fallback if nothing clean turns up
            while flank <= max_silent_search_flank:
                near = span_start - flank
                far  = span_end + flank
                silent_hits = find_silent_restriction_sites(
                    mutated_seq, orf_start, max(0, near), min(len(mutated_seq), far)
                )
                searched_flank = flank
                if silent_hits and best_unclean_hit is None:
                    best_unclean_hit = silent_hits[0]
                for candidate in silent_hits:
                    if _clean_candidate(candidate):
                        clean_hit = candidate
                        break
                if clean_hit is not None:
                    break
                if flank >= max_silent_search_flank:
                    break
                # max(flank, 1) guards against an infinite loop when flank
                # starts at 0 (0 * 3 == 0 forever).
                flank = min(max(flank, 1) * 3, max_silent_search_flank)

            hit = clean_hit or best_unclean_hit
            if hit is not None:
                diagnostic = DiagnosticInfo(
                    enzyme=hit["enzyme"],
                    effect=hit["effect"],
                    source="silent_mutation",
                    silent_aa_index=hit["aa_index"],
                    silent_original_codon=hit["original_codon"],
                    silent_new_codon=hit["new_codon"],
                    silent_changes=hit["changes"],
                )
                working_seq = _silent_mutation_seq(
                    mutated_seq, hit["position"], hit["new_codon"]
                )
                if searched_flank > silent_window_flank:
                    result.warnings.append(
                        f"Diagnostic site required widening the silent-mutation "
                        f"search to ±{searched_flank} nt (default ±{silent_window_flank} nt) "
                        f"to find {hit['enzyme']}."
                    )
                if clean_hit is None:
                    result.warnings.append(
                        f"No diagnostic enzyme within ±{searched_flank} nt was confirmed "
                        f"globally unique in the construct — {hit['enzyme']} was used as "
                        f"the best available option, but it may cut elsewhere unrelated "
                        f"to this mutation, making a simple digest unreliable as a "
                        f"pass/fail screen. Consider verifying by sequencing instead, or "
                        f"widen the search further."
                    )
            else:
                result.warnings.append(
                    f"No diagnostic restriction site found within ±{searched_flank} nt "
                    f"of the mutation (mutation itself gains/loses no site, and no "
                    f"1-codon silent option creates one nearby). "
                    "Restriction verification will be skipped."
                )

        result.diagnostic = diagnostic
        result.mutated_sequence = working_seq
        result.cut_site_diff = gained_lost_sites(sequence, working_seq)

        # ── Part 4: B/C primers ───────────────────────────────────────────────
        try:
            bc = design_bc_primers(working_seq, changed_pos, tm_range, min_overlap)
        except RuntimeError as exc:
            raise _PipelineError(f"B/C primer design failed: {exc}") from exc

        result.bc_result = bc
        result.overlap_seq = bc.overlap_seq
        result.overlap_tm  = bc.tm_overlap_fwd
        result.primer_B = PrimerInfo(
            sequence=bc.primer_b,
            tm=bc.tm_b,
            length=len(bc.primer_b),
            start=bc.b_start,
            end=bc.b_end,
        )
        result.primer_C = PrimerInfo(
            sequence=bc.primer_c,
            tm=bc.tm_c_full,
            tm_anneal=bc.tm_c_anneal,
            length=len(bc.primer_c),
            start=bc.c_start,
            end=bc.c_end,
        )

        # ── Part 5: A/D primers ───────────────────────────────────────────────
        # If the user supplied their own primer A and/or D sequence, that side
        # is located directly and the restriction-site search is skipped for
        # it. Any side left as None is still auto-designed as usual.
        need_auto_a = primer_A_seq is None
        need_auto_d = primer_D_seq is None

        near0, far0 = window_bp
        span = far0 - near0
        cur_window = window_bp
        result_a = result_d = None

        if need_auto_a or need_auto_d:
            # If no unique flanking site is found, widen the search window
            # outward and retry (no sequence changes — just looking further
            # away) before giving up.
            for attempt in range(max_ad_window_expansions + 1):
                try:
                    result_a, result_d = design_ad_primers(
                        working_seq,
                        b_start=bc.b_start,
                        c_end=bc.c_end,
                        tm_range=tm_range,
                        window_bp=cur_window,
                        target_tm=target_tm,
                        preferred=preferred_enzymes,
                    )
                except Exception as exc:
                    raise _PipelineError(f"A/D primer design failed: {exc}") from exc

                a_done = result_a.top is not None or not need_auto_a
                d_done = result_d.top is not None or not need_auto_d
                if a_done and d_done:
                    break
                if attempt == max_ad_window_expansions:
                    break
                cur_window = (near0, far0 + span * (attempt + 1))

        result.ad_result_a = result_a
        result.ad_result_d = result_d

        if primer_A_seq is not None:
            start_a, end_a, seq_a = _locate_user_primer(working_seq, primer_A_seq, "A")
            if start_a >= bc.overlap_start:
                raise _PipelineError(
                    f"Primer A (located at nt {start_a}-{end_a}) does not sit "
                    f"upstream of the B/C overlap (starts at nt {bc.overlap_start}) "
                    "— check that this is really the upstream/forward primer."
                )
            result.primer_A = PrimerInfo(
                sequence=seq_a, tm=simple_tm(seq_a), length=len(seq_a),
                start=start_a, end=end_a,
            )
            result.warnings.append(
                f"Primer A: using the supplied sequence (located at nt "
                f"{start_a}-{end_a} in the construct, Tm={simple_tm(seq_a):.0f}°C)."
            )
        elif result_a.top is None:
            result.warnings.append(
                f"No unique restriction site found for primer A even after "
                f"widening the search window up to {cur_window[0]}-{cur_window[1]} bp "
                "upstream. Try a longer construct, use --window to search a "
                "different range manually, or supply your own primer A sequence."
            )
        else:
            top_a = result_a.top
            result.primer_A = PrimerInfo(
                sequence=top_a.primer_seq,
                tm=top_a.primer_tm,
                length=len(top_a.primer_seq),
                start=top_a.primer_start,
                end=top_a.primer_end,
                enzyme=top_a.enzyme,
                cut_pos=top_a.cut_pos,
            )
            if top_a.extended_tm:
                result.warnings.append(
                    f"Primer A's Tm ({top_a.primer_tm:.0f}°C) is above the "
                    f"normal {tm_range[1]:.0f}°C target but was accepted because "
                    f"its 3' end is A/T — no length at this site hit the "
                    f"normal {tm_range[0]:.0f}-{tm_range[1]:.0f}°C range."
                )

        if primer_D_seq is not None:
            start_d, end_d, seq_d = _locate_user_primer(working_seq, primer_D_seq, "D")
            if end_d <= bc.overlap_end:
                raise _PipelineError(
                    f"Primer D (located at nt {start_d}-{end_d}) does not sit "
                    f"downstream of the B/C overlap (ends at nt {bc.overlap_end}) "
                    "— check that this is really the downstream/reverse primer."
                )
            result.primer_D = PrimerInfo(
                sequence=seq_d, tm=simple_tm(seq_d), length=len(seq_d),
                start=start_d, end=end_d,
            )
            result.warnings.append(
                f"Primer D: using the supplied sequence (located at nt "
                f"{start_d}-{end_d} in the construct, Tm={simple_tm(seq_d):.0f}°C)."
            )
        elif result_d.top is None:
            result.warnings.append(
                f"No unique restriction site found for primer D even after "
                f"widening the search window up to {cur_window[0]}-{cur_window[1]} bp "
                "downstream. Try a longer construct, use --window to search a "
                "different range manually, or supply your own primer D sequence."
            )
        else:
            top_d = result_d.top
            result.primer_D = PrimerInfo(
                sequence=top_d.primer_seq,
                tm=top_d.primer_tm,
                length=len(top_d.primer_seq),
                start=top_d.primer_start,
                end=top_d.primer_end,
                enzyme=top_d.enzyme,
                cut_pos=top_d.cut_pos,
            )
            if top_d.extended_tm:
                result.warnings.append(
                    f"Primer D's Tm ({top_d.primer_tm:.0f}°C) is above the "
                    f"normal {tm_range[1]:.0f}°C target but was accepted because "
                    f"its 3' end is A/T — no length at this site hit the "
                    f"normal {tm_range[0]:.0f}-{tm_range[1]:.0f}°C range."
                )

        if primer_A_seq is not None:
            cand_a = CandidateSite(
                enzyme="(user-supplied)", cut_pos=0, cut_pos_0=0,
                primer_seq=result.primer_A.sequence,
                primer_start=result.primer_A.start, primer_end=result.primer_A.end,
                primer_tm=result.primer_A.tm, tm_dist=0.0,
            )
            result_a = FlankingPrimerResult(
                label="A", window_start=result.primer_A.start,
                window_end=result.primer_A.end, candidates=[cand_a],
            )
            result.ad_result_a = result_a

        if primer_D_seq is not None:
            cand_d = CandidateSite(
                enzyme="(user-supplied)", cut_pos=0, cut_pos_0=0,
                primer_seq=result.primer_D.sequence,
                primer_start=result.primer_D.start, primer_end=result.primer_D.end,
                primer_tm=result.primer_D.tm, tm_dist=0.0,
            )
            result_d = FlankingPrimerResult(
                label="D", window_start=result.primer_D.start,
                window_end=result.primer_D.end, candidates=[cand_d],
            )
            result.ad_result_d = result_d

        # ── Part 6: assembly and verification ─────────────────────────────────
        if result.primer_A is None or result.primer_D is None:
            result.warnings.append(
                "Skipping assembly verification because primer A or D is missing."
            )
        else:
            try:
                report = run_full_verification(
                    original_seq=sequence,
                    mutated_seq=working_seq,
                    orf_start=orf_start,
                    aa_position=positions,
                    original_aa=original_aas,
                    new_aa=new_aas,
                    original_codon=result.original_codon,
                    new_codon=result.new_codon,
                    changed_positions=changed_pos,
                    bc=bc,
                    result_a=result_a,
                    result_d=result_d,
                    diagnostic_enzyme=diagnostic.enzyme if diagnostic else None,
                    diagnostic_expected_present=(
                        diagnostic.effect == "gained" if diagnostic else True
                    ),
                    mutation_label=mutation_label,
                )
            except Exception as exc:
                raise _PipelineError(f"Verification failed: {exc}") from exc

            result.report = report
            result.frag_ab_bp = report.frag_ab_len
            result.frag_cd_bp = report.frag_cd_len
            result.frag_ad_bp = report.assembled_len
            result.frag_ab_seq = report.frag_ab_seq
            result.frag_cd_seq = report.frag_cd_seq
            result.frag_ad_seq = report.assembled_seq

            mc = report.mutation_check
            result.translation_passed = mc.passed
            result.translation_detail = mc.message

            rc = report.restriction_check
            if rc is not None:
                result.restriction_passed = rc.passed
                result.restriction_detail = rc.message
            else:
                result.restriction_passed = None
                result.restriction_detail = "No diagnostic site — check skipped"

            result.overall_passed = mc.passed and (rc is None or rc.passed)

    except _PipelineError as exc:
        result.errors.append(str(exc))
        result.overall_passed = False

    return result


def result_to_dict(r: PipelineResult) -> dict[str, Any]:
    """
    Serialize a PipelineResult to a plain dict (JSON-friendly; no Bio objects).
    """
    def _primer(p: PrimerInfo | None) -> dict | None:
        if p is None:
            return None
        d = dict(
            sequence=p.sequence,
            tm=p.tm,
            length=p.length,
            start=p.start,
            end=p.end,
        )
        if p.enzyme is not None:
            d["enzyme"] = p.enzyme
            d["cut_pos"] = p.cut_pos
        if p.tm_anneal is not None:
            d["tm_anneal"] = p.tm_anneal
        return d

    def _diag(d: DiagnosticInfo | None) -> dict | None:
        if d is None:
            return None
        out = dict(enzyme=d.enzyme, effect=d.effect, source=d.source)
        if d.source == "silent_mutation":
            out.update(
                silent_aa_index=d.silent_aa_index,
                silent_original_codon=d.silent_original_codon,
                silent_new_codon=d.silent_new_codon,
                silent_changes=d.silent_changes,
            )
        return out

    return dict(
        mutation=r.mutation_label,
        original_codon=r.original_codon,
        new_codon=r.new_codon,
        orf_start=r.orf_start_detected,
        mutation_nt_position=r.mutation_nt_position,
        changed_positions=r.changed_positions,
        mutations=[
            dict(position=p, original_aa=oa, new_aa=na, original_codon=oc, new_codon=nc)
            for p, oa, na, oc, nc in zip(
                r.mutation_positions, r.original_aas, r.new_aas,
                r.original_codons, r.new_codons,
            )
        ],
        mutated_sequence=r.mutated_sequence,
        original_sequence=r.original_sequence,
        primers=dict(
            A=_primer(r.primer_A),
            B=_primer(r.primer_B),
            C=_primer(r.primer_C),
            D=_primer(r.primer_D),
        ),
        overlap=dict(sequence=r.overlap_seq, tm=r.overlap_tm),
        diagnostic=_diag(r.diagnostic),
        cut_site_diff=r.cut_site_diff,
        fragment_lengths=dict(ab=r.frag_ab_bp, cd=r.frag_cd_bp, ad=r.frag_ad_bp),
        fragment_sequences=dict(ab=r.frag_ab_seq, cd=r.frag_cd_seq, ad=r.frag_ad_seq),
        verification=dict(
            translation=dict(passed=r.translation_passed, detail=r.translation_detail),
            restriction=dict(passed=r.restriction_passed, detail=r.restriction_detail),
            overall=r.overall_passed,
        ),
        warnings=r.warnings,
        errors=r.errors,
    )
