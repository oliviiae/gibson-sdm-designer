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
from primer_ad import design_ad_primers, FlankingPrimerResult, PREFERRED_ENZYMES
from restriction_utils import gained_lost_sites, find_silent_restriction_sites, scan_sites
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
    mutation_label: str          # e.g. "K15E"
    orf_start: int
    original_codon: str
    new_codon: str
    changed_positions: list[int]
    orf_start_detected: int = 0       # the orf_start actually used (auto or supplied)
    mutation_nt_position: int = 0     # 0-based nucleotide position of first changed base

    # ── Primers ───────────────────────────────────────────────────────────
    primer_A: PrimerInfo | None = None
    primer_B: PrimerInfo | None = None
    primer_C: PrimerInfo | None = None
    primer_D: PrimerInfo | None = None
    overlap_seq: str = ""
    overlap_tm: float = 0.0

    # ── Diagnostic restriction site ───────────────────────────────────────
    diagnostic: DiagnosticInfo | None = None

    # ── Fragment sizes ────────────────────────────────────────────────────
    frag_ab_bp: int = 0
    frag_cd_bp: int = 0
    frag_ad_bp: int = 0    # total assembled product

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
    target_position: int,
    original_aa: str,
    new_aa: str,
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
) -> PipelineResult:
    """
    Full Gibson SDM primer design pipeline (Parts 1–6).

    Parameters
    ----------
    sequence         : DNA sequence of the full construct (any case; IUPAC A/C/G/T)
    target_position  : 1-based amino acid number to mutate
    original_aa      : single-letter AA expected at that position (confirms identity)
    new_aa           : desired amino acid
    orf_start        : 0-based position where the ORF begins (the A in ATG).
                       If None, the sequence is scanned automatically and the
                       unique ATG that places original_aa at target_position is used.
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

    Returns
    -------
    PipelineResult — always returned, even on error.
    Check result.errors for fatal failures; result.warnings for soft ones.
    """
    sequence = sequence.upper().strip()
    original_aa = original_aa.upper()
    new_aa = new_aa.upper()
    mutation_label = f"{original_aa}{target_position}{new_aa}"

    result = PipelineResult(
        mutation_label=mutation_label,
        orf_start=orf_start or 0,
        original_codon="",
        new_codon="",
        changed_positions=[],
    )

    try:
        # ── Step 0: resolve orf_start and validate inputs ────────────────────
        if target_position < 1:
            raise _PipelineError("target_position must be ≥ 1.")

        if orf_start is None:
            # Auto-detect: find every ATG where position target_position is original_aa
            candidates = find_orf_start(sequence, original_aa, target_position)
            if len(candidates) == 0:
                raise _PipelineError(
                    f"Could not find an ATG in the sequence where amino acid "
                    f"{target_position} is {original_aa}. "
                    "Check that the mutation label matches the sequence, "
                    "or supply --orf-start manually."
                )
            if len(candidates) > 1:
                positions_str = ", ".join(str(c) for c in candidates)
                raise _PipelineError(
                    f"Found {len(candidates)} ATGs where position {target_position} "
                    f"is {original_aa} (at nt positions {positions_str}). "
                    "Sequence is ambiguous — supply --orf-start to specify which one."
                )
            orf_start = candidates[0]
            result.orf_start = orf_start

        if orf_start < 0 or orf_start >= len(sequence):
            raise _PipelineError(
                f"orf_start {orf_start} is outside sequence length {len(sequence)}."
            )
        codon_abs = orf_start + (target_position - 1) * 3
        if codon_abs + 3 > len(sequence):
            raise _PipelineError(
                f"Codon for position {target_position} extends past the end "
                f"of the sequence (need pos {codon_abs}–{codon_abs+2}, "
                f"sequence is {len(sequence)} bp)."
            )
        # Confirm original AA
        actual_codon = sequence[codon_abs: codon_abs + 3]
        actual_aa = str(Seq(actual_codon).translate())
        if actual_aa != original_aa:
            raise _PipelineError(
                f"Position {target_position}: expected {original_aa} "
                f"but found {actual_aa} (codon {actual_codon}). "
                "Check orf_start and target_position."
            )

        result.orf_start_detected = orf_start

        # ── Part 1: codon mutation ────────────────────────────────────────────
        try:
            mutated_seq, changed_pos, orig_codon, new_codon = find_codon_and_mutate(
                sequence, orf_start, target_position, new_aa
            )
        except ValueError as exc:
            raise _PipelineError(f"Codon mutation failed: {exc}") from exc

        result.original_codon = orig_codon
        result.new_codon = new_codon
        result.changed_positions = changed_pos
        result.mutation_nt_position = changed_pos[0] if changed_pos else codon_abs

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
        else:
            # Protocol step 1a: "change the nucleotide again" — before
            # resorting to an adjacent-codon silent mutation, try OTHER
            # codons that still encode the same target amino acid (there is
            # often more than one, and the fewest-change pick above isn't
            # necessarily the one that creates/destroys a usable site).
            for alt_codon in ranked_codon_options(orig_codon, new_aa):
                alt_seq, alt_changed_pos, _, _ = apply_codon(
                    sequence, orf_start, target_position, alt_codon
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
                        f"Used alternate codon {new_codon} for {new_aa} (instead of "
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
            while flank <= max_silent_search_flank:
                near = codon_abs - flank
                far  = codon_abs + 3 + flank
                silent_hits = find_silent_restriction_sites(
                    mutated_seq, orf_start, max(0, near), min(len(mutated_seq), far)
                )
                searched_flank = flank
                if silent_hits:
                    break
                if flank >= max_silent_search_flank:
                    break
                # max(flank, 1) guards against an infinite loop when flank
                # starts at 0 (0 * 3 == 0 forever).
                flank = min(max(flank, 1) * 3, max_silent_search_flank)

            if silent_hits:
                # Same uniqueness concern as the direct-mutation diagnostic
                # above: a candidate enzyme found via the local-window search
                # could still have other pre-existing sites elsewhere in the
                # full construct, making it a useless gained/lost signal on a
                # real digest. Prefer the first candidate (already sorted by
                # fewest nt changes) that is globally clean; fall back to the
                # raw first candidate if none are — Part 6 verification will
                # still catch and flag it as a FAIL rather than a false PASS.
                hit = silent_hits[0]
                for candidate in silent_hits:
                    cand_seq = _silent_mutation_seq(
                        mutated_seq, candidate["position"], candidate["new_codon"]
                    )
                    cand_after = scan_sites(cand_seq)
                    enz = candidate["enzyme"]
                    if candidate["effect"] == "gained":
                        clean = (
                            len(after_sites.get(enz, [])) == 0
                            and len(cand_after.get(enz, [])) == len(candidate["site_positions"])
                        )
                    else:  # "lost"
                        clean = (
                            len(cand_after.get(enz, [])) == 0
                            and len(after_sites.get(enz, [])) == len(candidate["site_positions"])
                        )
                    if clean:
                        hit = candidate
                        break
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
            else:
                result.warnings.append(
                    f"No diagnostic restriction site found within ±{searched_flank} nt "
                    f"of the mutation (mutation itself gains/loses no site, and no "
                    f"1-codon silent option creates one nearby). "
                    "Restriction verification will be skipped."
                )

        result.diagnostic = diagnostic

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

        # Primer B's start is fixed while the overlap's start can shift a few
        # bases later to satisfy the G/C-terminus + Tm-range constraints
        # together — when that happens, B ends up longer than the overlap
        # alone and its own Tm can run above the target range. The overlap
        # itself is still separately verified in range; this is just a
        # heads-up that primer B as a whole runs hot.
        tm_lo, tm_hi = tm_range
        if bc.tm_b > tm_hi:
            result.warnings.append(
                f"Primer B's full Tm ({bc.tm_b:.0f}°C) runs above the target "
                f"{tm_lo:.0f}-{tm_hi:.0f}°C range because it had to extend a few "
                f"bases past the overlap to reach a valid G/C-terminated overlap "
                f"start. The overlap itself (Tm={bc.tm_overlap_fwd:.0f}°C) is "
                f"still in range — this only affects primer B's own annealing Tm."
            )

        # Primer C's Tm is walked over [overlap + annealing tail] combined, so
        # the annealing-only portion (the part that actually binds fresh
        # template each cycle) can land well outside 48-54°C as a byproduct,
        # even though the combined walk landed in range. A low anneal Tm is a
        # real practical concern (weak/nonspecific priming), so flag it.
        if bc.tm_c_anneal < tm_lo:
            result.warnings.append(
                f"Primer C's unique annealing region (excluding the shared "
                f"overlap) has Tm {bc.tm_c_anneal:.0f}°C, below the target "
                f"{tm_lo:.0f}-{tm_hi:.0f}°C range. This is the portion that "
                f"actually binds fresh template each PCR cycle — a low Tm here "
                f"risks weak or nonspecific priming for the 'cd' fragment. "
                f"This is a side effect of where primer C's endpoint landed "
                f"relative to the overlap for this specific sequence, not "
                f"something --min-overlap can reliably fix (raising it can "
                f"make this worse; lowering it below 16nt would violate the "
                f"minimum overlap length). Consider a lower annealing "
                f"temperature or gradient PCR for the 'cd' fragment."
            )
        elif bc.tm_c_anneal > tm_hi:
            result.warnings.append(
                f"Primer C's unique annealing region (excluding the shared "
                f"overlap) has Tm {bc.tm_c_anneal:.0f}°C, above the target "
                f"{tm_lo:.0f}-{tm_hi:.0f}°C range."
            )

        # ── Part 5: A/D primers ───────────────────────────────────────────────
        # If no unique flanking site is found, widen the search window
        # outward and retry (no sequence changes — just looking further away)
        # before giving up.
        near0, far0 = window_bp
        span = far0 - near0
        cur_window = window_bp
        result_a = result_d = None
        windows_tried = [cur_window]

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

            if result_a.top is not None and result_d.top is not None:
                break
            if attempt == max_ad_window_expansions:
                break
            cur_window = (near0, far0 + span * (attempt + 1))
            windows_tried.append(cur_window)

        if len(windows_tried) > 1 and (result_a.top or result_d.top):
            result.warnings.append(
                f"Default flanking window {window_bp[0]}-{window_bp[1]} bp had no "
                f"unique site; widened to {cur_window[0]}-{cur_window[1]} bp to find one "
                f"(no sequence was changed — only the search range)."
            )

        result.ad_result_a = result_a
        result.ad_result_d = result_d

        if result_a.top is None:
            result.warnings.append(
                f"No unique restriction site found for primer A even after "
                f"widening the search window up to {cur_window[0]}-{cur_window[1]} bp "
                "upstream. Try a longer construct, or use --window to search a "
                "different range manually."
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

        if result_d.top is None:
            result.warnings.append(
                f"No unique restriction site found for primer D even after "
                f"widening the search window up to {cur_window[0]}-{cur_window[1]} bp "
                "downstream. Try a longer construct, or use --window to search a "
                "different range manually."
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
                    aa_position=target_position,
                    original_aa=original_aa,
                    new_aa=new_aa,
                    original_codon=orig_codon,
                    new_codon=new_codon,
                    changed_positions=changed_pos,
                    bc=bc,
                    result_a=result_a,
                    result_d=result_d,
                    diagnostic_enzyme=diagnostic.enzyme if diagnostic else None,
                    diagnostic_expected_present=(
                        diagnostic.effect == "gained" if diagnostic else True
                    ),
                )
            except Exception as exc:
                raise _PipelineError(f"Verification failed: {exc}") from exc

            result.report = report
            result.frag_ab_bp = report.frag_ab_len
            result.frag_cd_bp = report.frag_cd_len
            result.frag_ad_bp = report.assembled_len

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
        primers=dict(
            A=_primer(r.primer_A),
            B=_primer(r.primer_B),
            C=_primer(r.primer_C),
            D=_primer(r.primer_D),
        ),
        overlap=dict(sequence=r.overlap_seq, tm=r.overlap_tm),
        diagnostic=_diag(r.diagnostic),
        fragment_lengths=dict(ab=r.frag_ab_bp, cd=r.frag_cd_bp, ad=r.frag_ad_bp),
        verification=dict(
            translation=dict(passed=r.translation_passed, detail=r.translation_detail),
            restriction=dict(passed=r.restriction_passed, detail=r.restriction_detail),
            overall=r.overall_passed,
        ),
        warnings=r.warnings,
        errors=r.errors,
    )
