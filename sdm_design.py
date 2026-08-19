#!/usr/bin/env python3
"""
sdm_design.py — Gibson site-directed mutagenesis primer designer.

Usage
-----
  python sdm_design.py construct.fasta --orf-start 0 --mutation K255E
  python sdm_design.py construct.fasta --orf-start 0 K 255 E
  python sdm_design.py construct.fasta --orf-start 0 --mutation K255E --json

Options
-------
  FASTA_FILE            Path to a FASTA file containing the construct sequence.
                        Multi-record files use the first record.
  --orf-start INT       0-based position of the first base of the ORF (the A
                        in the ATG start codon).  Required.
  --mutation STR        Mutation in <original><position><new> format, e.g. K255E.
                        For a combined double mutation sharing one primer set,
                        join two with "+": K255E+R300A. Only close-together
                        positions can realistically share one B/C overlap.
                        Alternatively supply --orig-aa, --position, --new-aa
                        (single mutation only).
  --orig-aa AA          Single-letter original amino acid (e.g. K).
  --position INT        1-based amino acid position.
  --new-aa AA           Single-letter desired amino acid (e.g. E).
  --window INT INT      Near and far bounds (bp) of the flanking primer search
                        window, applied on EACH side independently. Default
                        350 500 (~700-1000bp total between primer A and D).
  --tm-min FLOAT        Lower Tm bound (°C).  Default: 48.
  --tm-max FLOAT        Upper Tm bound (°C).  Default: 54.
  --json                Print the result as a JSON object instead of the
                        formatted report.
  --all-candidates      Show all A/D primer candidates, not just the top pick.
"""

import argparse
import json
import sys
import textwrap

from Bio import SeqIO

from pipeline import (
    design_mutation_primers, parse_mutation_label, parse_mutation_labels, result_to_dict,
)
from assembly import print_report, translate_orf
from primer_ad import NEB_CATALOG
from restriction_utils import scan_sites


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sdm_design.py",
        description="Design Gibson primers for a point mutation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Find all K positions (to see where K→E could be made):
              python sdm_design.py construct.fasta --find K E

              # Design primers once you know the position:
              python sdm_design.py construct.fasta --mutation K255E

              # If the tool says the sequence is ambiguous, add --orf-start:
              python sdm_design.py construct.fasta --mutation K255E --orf-start 149
        """),
    )
    p.add_argument("fasta_file", metavar="SEQUENCE_FILE", nargs="?", default=None,
                   help="FASTA (.fasta/.fa) or .xdna file containing the "
                        "construct sequence. Format is auto-detected from "
                        "the file extension.")
    p.add_argument("--orf-start", type=int, default=None,
                   help="0-based position of the ORF start codon (ATG). "
                        "If omitted, the sequence is scanned automatically.")

    p.add_argument("--find", nargs=2, metavar=("FROM_AA", "TO_AA"),
                   help="Find positions of FROM_AA where primer design will "
                        "actually succeed, e.g. --find K E. Positions that "
                        "would fail (too close to sequence ends, no diagnostic "
                        "site, etc) are filtered out automatically.")
    p.add_argument("--show-all", action="store_true",
                   help="With --find, also list positions that were filtered "
                        "out, with the reason for each.")

    mut_group = p.add_argument_group("mutation specification (use --mutation OR all three)")
    mut_group.add_argument("--mutation", metavar="e.g. K255E or K255E+R300A",
                           help="Compact mutation label. Join two with '+' "
                                "for a combined double mutation sharing one "
                                "primer set.")
    mut_group.add_argument("--orig-aa",  metavar="AA")
    mut_group.add_argument("--position", type=int, metavar="INT")
    mut_group.add_argument("--new-aa",   metavar="AA")

    p.add_argument("--window", nargs=2, type=int, metavar=("NEAR", "FAR"),
                   default=[350, 500],
                   help="Flanking primer search window in bp, applied on EACH "
                        "side independently (upstream of B for primer A, "
                        "downstream of C for primer D). Default 350 500 "
                        "targets ~700-1000bp total between primer A and D "
                        "(per protocol: 'two sites around 700-1000bp apart'), "
                        "not 700-1000bp on each side.")
    p.add_argument("--tm-min", type=float, default=48.0,
                   help="Minimum Tm (°C) for primer design. Default: 48.")
    p.add_argument("--tm-max", type=float, default=54.0,
                   help="Maximum Tm (°C) for primer design. Default: 54.")
    p.add_argument("--min-overlap", type=int, default=16,
                   help="Minimum B/C overlap length in bp. Default: 16 "
                        "(the protocol's stated minimum — do not go below "
                        "this). Raising it does not fix a low primer-C "
                        "annealing Tm warning and can make it worse; it's "
                        "mainly useful if you want a longer/stronger overlap "
                        "for its own sake.")
    p.add_argument("--primer-a-seq", metavar="SEQ", default=None,
                   help="Use your own primer A sequence instead of having the "
                        "tool search for a restriction site. Located in the "
                        "construct as given or as its reverse complement.")
    p.add_argument("--primer-d-seq", metavar="SEQ", default=None,
                   help="Use your own primer D sequence instead of having the "
                        "tool search for a restriction site. Located in the "
                        "construct as given or as its reverse complement.")
    p.add_argument("--json", action="store_true",
                   help="Output results as JSON.")
    p.add_argument("--all-candidates", action="store_true",
                   help="Print all A/D primer candidates, not just the top pick.")
    p.add_argument("--neb-info", action="store_true",
                   help="Print NEB catalog numbers and notes for all preferred "
                        "restriction enzymes, then exit.")
    p.add_argument("--output", "-o", metavar="FILE",
                   help="Also save the report to FILE (same content as printed "
                        "to the console — JSON if --json is set, otherwise the "
                        "formatted report).")
    return p


def _resolve_mutations(args: argparse.Namespace) -> list[tuple[str, int, str]]:
    """
    Return a list of (original_aa, position, new_aa) tuples — one entry for
    a single mutation, multiple for a combined "+"-joined multi-mutation
    (e.g. --mutation K255E+R300A) — or exit with an error.
    """
    if args.mutation:
        try:
            return parse_mutation_labels(args.mutation)
        except ValueError as exc:
            sys.exit(f"ERROR: {exc}")

    if args.orig_aa and args.position and args.new_aa:
        return [(args.orig_aa.upper(), args.position, args.new_aa.upper())]

    sys.exit(
        "ERROR: Provide either --mutation (e.g. K255E, or K255E+R300A for a "
        "combined double mutation) or all three of --orig-aa, --position, --new-aa."
    )


def _read_fasta(path: str) -> str:
    """Read the first record from a FASTA file. Exits on any error."""
    try:
        records = list(SeqIO.parse(path, "fasta"))
    except FileNotFoundError:
        sys.exit(f"ERROR: File not found: {path}")
    except Exception as exc:
        sys.exit(f"ERROR reading FASTA: {exc}")

    if not records:
        sys.exit(f"ERROR: No sequences found in {path}")
    if len(records) > 1:
        print(
            f"WARNING: {len(records)} records in {path}; using the first "
            f"({records[0].id}, {len(records[0])} bp).",
            file=sys.stderr,
        )
    seq = str(records[0].seq).upper()
    if not seq:
        sys.exit(f"ERROR: First FASTA record is empty.")
    non_dna = set(seq) - set("ACGTN")
    if non_dna:
        print(
            f"WARNING: sequence contains non-standard bases "
            f"{sorted(non_dna)[:6]}; these will be treated as N.",
            file=sys.stderr,
        )
    return seq


def _read_xdna(path: str) -> str:
    """Read a .xdna file using the heuristic DNA-run extractor. Exits on error."""
    from xdna_utils import read_xdna_sequence, print_verification

    try:
        result = read_xdna_sequence(path)
    except FileNotFoundError:
        sys.exit(f"ERROR: File not found: {path}")
    except ValueError as exc:
        sys.exit(
            f"ERROR reading .xdna file: {exc}\n"
            f"Safer alternative: open the file in SnapGene / Serial Cloner / ApE "
            f"and export it as FASTA, then re-run with that file."
        )

    print_verification(result, path)
    return result.sequence


def _read_sequence_file(path: str) -> str:
    """Dispatch to the right reader based on file extension."""
    if path.lower().endswith(".xdna"):
        return _read_xdna(path)
    return _read_fasta(path)


# ---------------------------------------------------------------------------
# Formatted console output (non-JSON path)
# ---------------------------------------------------------------------------

W = 64   # column width for the report

def _hr(char="─"):
    return char * W


def _print_candidates(label: str, res, show_all: bool):
    cands = res.candidates
    if not cands:
        print(f"  No unique sites found in window [{res.window_start},{res.window_end}).")
        return
    limit = len(cands) if show_all else min(10, len(cands))
    print(f"  {'#':<4} {'Enzyme':<14} {'Cut (1-based)':<15} "
          f"{'Tm':>6}  Primer sequence")
    print(f"  {'-'*4} {'-'*14} {'-'*15} {'-'*6}  {'-'*30}")
    for i, c in enumerate(cands[:limit], 1):
        marker = " ◀ top" if i == 1 else ""
        print(f"  {i:<4} {c.enzyme:<14} {c.cut_pos:<15} "
              f"{c.primer_tm:>5.0f}°C  5'-{c.primer_seq}-3'{marker}")
    if not show_all and len(cands) > limit:
        print(f"  … and {len(cands)-limit} more (use --all-candidates to show all)")


def _print_seq_with_cuts(
    seq: str, frag_start: int, cuts: list, unit: str = "this fragment",
    mutation_ranges: list[tuple[int, int, str]] | None = None,
):
    """
    Print seq with a caret marking each restriction enzyme's cut position.
    cuts: list of (cut_pos_1based_in_full_construct, enzyme_name_or_None).
    frag_start: 0-based absolute position where seq begins in the full
    construct (0 when seq itself IS the full construct).

    mutation_ranges: optional list of (start_abs, end_abs, label) — 0-based
    absolute [start, end) spans (e.g. a mutated codon) marked with '~' on a
    second marker line, so a mutation's location can be shown alongside the
    restriction cut sites it affects.
    """
    print(f"      {seq}")
    markers = [" "] * len(seq)
    labels = []
    for cut_pos, label in cuts:
        if cut_pos is None:
            continue
        rel = (cut_pos - 1) - frag_start
        if 0 <= rel < len(seq):
            markers[rel] = "^"
            labels.append((rel, label))
    if labels:
        print(f"      {''.join(markers)}".rstrip())
        for rel, label in sorted(labels):
            print(f"      {' ' * rel}└─ {label} cuts here (nt {rel} in {unit})")

    range_markers = [" "] * len(seq)
    range_labels = []
    for start_abs, end_abs, label in (mutation_ranges or []):
        rel_start = max(0, start_abs - frag_start)
        rel_end = min(len(seq), end_abs - frag_start)
        if rel_start >= rel_end:
            continue
        for idx in range(rel_start, rel_end):
            range_markers[idx] = "~"
        range_labels.append((rel_start, label))
    if range_labels:
        print(f"      {''.join(range_markers)}".rstrip())
        for rel, label in sorted(range_labels):
            print(f"      {' ' * rel}└─ {label} (nt {rel} in {unit})")


def _collect_all_cut_sites(result) -> list[tuple[int | None, str | None]]:
    """
    Every restriction cut site actually relevant to this design: primer A's
    site, primer D's site, and the diagnostic enzyme's site(s) — the one
    used to verify the clone by digest, easy to overlook since it isn't "A"
    or "D". Only meaningful when the diagnostic site was GAINED (present in
    the mutated sequence); a LOST site has nothing to mark since it's absent.
    """
    cuts: list[tuple[int | None, str | None]] = []
    if result.primer_A:
        cuts.append((result.primer_A.cut_pos, f"{result.primer_A.enzyme} (primer A)"))
    if result.primer_D:
        cuts.append((result.primer_D.cut_pos, f"{result.primer_D.enzyme} (primer D)"))
    if result.diagnostic and result.diagnostic.effect == "gained" and result.mutated_sequence:
        positions = scan_sites(result.mutated_sequence).get(result.diagnostic.enzyme, [])
        for pos in positions:
            cuts.append((pos, f"{result.diagnostic.enzyme} (diagnostic)"))
    return cuts


def _bracket_spans(seq: str, spans: list[tuple[int, int]]) -> str:
    """Wrap each [start, end) span in seq with [ ]. spans may be unsorted/adjacent."""
    out = seq
    for start, end in sorted(spans, key=lambda s: -s[0]):
        out = out[:start] + "[" + out[start:end] + "]" + out[end:]
    return out


def _bracket_spans_styled(seq: str, spans: list[tuple[int, int, str, str]]) -> str:
    """
    Like _bracket_spans, but each span carries its own (open, close) bracket
    pair — e.g. "[","]" for the primary mutation vs "{","}" for a silent
    diagnostic mutation, so the two are visually distinguishable in plain
    text. spans may be unsorted/adjacent.
    """
    out = seq
    for start, end, ochar, cchar in sorted(spans, key=lambda s: -s[0]):
        out = out[:start] + ochar + out[start:end] + cchar + out[end:]
    return out


def _print_mutated_region(result, aa_flank: int = 15):
    """
    Print the mutated ORF's DNA and translated protein for a window around
    the mutation(s), with the changed codon(s)/residue(s) in [brackets]. If
    a silent diagnostic mutation was added, its codon/residue is included in
    the window (widening it if needed) and shown in {braces}, since it's a
    real edit to the ordered primers even though it's not the target change.
    """
    if not result.mutated_sequence or not result.mutation_positions:
        return

    orf_start = result.orf_start_detected
    mutated_seq = result.mutated_sequence
    positions = result.mutation_positions  # 1-based aa positions

    try:
        protein = translate_orf(mutated_seq, orf_start)
    except ValueError:
        return
    if not protein:
        return

    aa_min, aa_max = min(positions), max(positions)

    silent_aa = None
    d = result.diagnostic
    if d is not None and d.source == "silent_mutation" and d.silent_aa_index is not None:
        silent_aa = d.silent_aa_index

    win_min = min(aa_min, silent_aa) if silent_aa is not None else aa_min
    win_max = max(aa_max, silent_aa) if silent_aa is not None else aa_max
    win_start = max(1, win_min - aa_flank)
    win_end = min(len(protein), win_max + aa_flank)

    dna_start = orf_start + (win_start - 1) * 3
    dna_end = orf_start + win_end * 3
    dna_window = mutated_seq[dna_start:dna_end]
    protein_window = protein[win_start - 1: win_end]

    codon_spans = [
        ((pos - win_start) * 3, (pos - win_start) * 3 + 3, "[", "]")
        for pos in positions if win_start <= pos <= win_end
    ]
    residue_spans = [
        (pos - win_start, pos - win_start + 1, "[", "]")
        for pos in positions if win_start <= pos <= win_end
    ]
    if silent_aa is not None and win_start <= silent_aa <= win_end:
        codon_spans.append(
            ((silent_aa - win_start) * 3, (silent_aa - win_start) * 3 + 3, "{", "}")
        )
        residue_spans.append(
            (silent_aa - win_start, silent_aa - win_start + 1, "{", "}")
        )

    note = "target mutation in [brackets]"
    if silent_aa is not None:
        note += ", silent diagnostic mutation in {braces}"

    print(f"\n  {_hr()}")
    print(f"  Mutated region  (aa {win_start}-{win_end}, {note})")
    print(f"    DNA      5'-{_bracket_spans_styled(dna_window, codon_spans)}-3'")
    print(f"    Protein     {_bracket_spans_styled(protein_window, residue_spans)}")


def _print_formatted(result, show_all_candidates: bool):
    print()
    print(_hr("═"))
    print(f"  Gibson Primer Designer — {result.mutation_label}")
    print(_hr("═"))

    # Warnings / errors at the top
    for w in result.warnings:
        print(f"\n  ⚠  {w}")
    for e in result.errors:
        print(f"\n  ✗  ERROR: {e}")
    if result.errors:
        print()
        return

    # Mutation
    print(f"\n  Mutation        {result.mutation_label}")
    print(f"  Codon           {result.original_codon} → {result.new_codon}")
    print(f"  ORF start       nt {result.orf_start_detected}  (0-based)")
    print(f"  Mutation site   nt {result.mutation_nt_position}  (0-based, first changed base)")

    # Mutated region: DNA + translation, with the changed codon(s)/residue(s)
    # bracketed. Shown as a local window (not the whole ORF) since a real
    # gene can be thousands of bp — the useful check is what's right around
    # the edit, not scrolling through the entire coding sequence.
    _print_mutated_region(result)

    # Diagnostic
    print(f"\n  {_hr()}")
    print(f"  Diagnostic restriction site")
    if result.diagnostic is None:
        print("    None found — restriction verification unavailable")
    else:
        d = result.diagnostic
        src = "(from mutation)" if d.source == "mutation" else \
              (f"(silent mutation: aa{d.silent_aa_index} "
               f"{d.silent_original_codon}→{d.silent_new_codon}, "
               f"{d.silent_changes} nt change)")
        print(f"    Enzyme : {d.enzyme}  —  site {d.effect.upper()}  {src}")

    # Cutting pattern diff: every NEB enzyme gained/lost anywhere in the
    # construct, wild type vs mutant — not just the one enzyme picked above
    # as the diagnostic, so you can see the full restriction-map impact of
    # the edit (e.g. to sanity-check nothing unexpected also changed).
    diff = result.cut_site_diff or {}
    gained = diff.get("gained", {})
    lost = diff.get("lost", {})
    if gained or lost:
        print(f"\n  {_hr()}")
        print(f"  Cutting pattern diff  (wild type vs mutant)")
        for enz in sorted(gained):
            positions = ", ".join(str(p) for p in gained[enz])
            print(f"    + {enz:<10} gained at nt {positions} (mutant only)")
        for enz in sorted(lost):
            positions = ", ".join(str(p) for p in lost[enz])
            print(f"    − {enz:<10} lost at nt {positions} (wild type only)")

        # Mutation/silent-mutation codon ranges, shown on both views (same
        # coordinates, since substitutions don't shift length) so it's
        # clear WHERE relative to the cut-site diff the actual edit(s) are.
        orf_start = result.orf_start_detected
        mutation_ranges = []
        for pos in result.mutation_positions or []:
            codon_start = orf_start + (pos - 1) * 3
            mutation_ranges.append((codon_start, codon_start + 3, "target mutation"))
        diag = result.diagnostic
        if diag is not None and diag.source == "silent_mutation" and diag.silent_aa_index is not None:
            codon_start = orf_start + (diag.silent_aa_index - 1) * 3
            mutation_ranges.append((codon_start, codon_start + 3, "silent diagnostic mutation"))

        # Direct stacked comparison: wild type directly above mutant, each
        # with its diff sites (^) and the mutation site(s) (~) marked, at
        # matching coordinates, so the two lines can be read one against
        # the other.
        if result.original_sequence and result.mutated_sequence:
            wt_cuts = [(p, e) for e, ps in lost.items() for p in ps]
            mut_cuts = [(p, e) for e, ps in gained.items() for p in ps]
            print(f"\n    Wild type  (- = site lost, ~ = mutation site)")
            _print_seq_with_cuts(result.original_sequence, 0, wt_cuts, unit="wild type",
                                  mutation_ranges=mutation_ranges)
            print(f"\n    Mutant  (+ = site gained, ~ = mutation site)")
            _print_seq_with_cuts(result.mutated_sequence, 0, mut_cuts, unit="mutant",
                                  mutation_ranges=mutation_ranges)

    # Primers
    print(f"\n  {_hr()}")
    print(f"  Primers  (5'→3')")

    def _prow(lbl, seq, tm, note=""):
        tag = f"Tm={tm:.0f}°C"
        print(f"    {lbl}  {tag:<10}  {seq}  {note}")

    if result.primer_A:
        pa = result.primer_A
        if pa.enzyme is None:
            _prow("A", pa.sequence, pa.tm, "[user-supplied]")
        else:
            neb = NEB_CATALOG.get(pa.enzyme or "", {})
            neb_tag = f"NEB {neb['hf'] or neb['cat']} (HF)" if neb.get("hf") else \
                      (f"NEB {neb['cat']}" if neb else "")
            warn_tag = f"  ⚠ {neb['note']}" if neb.get("note") else ""
            _prow("A", pa.sequence, pa.tm,
                  f"[{pa.enzyme}  {neb_tag}  cut {pa.cut_pos}]{warn_tag}")
    else:
        print("    A  (not found)")

    if result.primer_B:
        pb = result.primer_B
        tm_note = (f"Tm={pb.tm:.0f}°C full / "
                   f"{pb.tm_anneal:.0f}°C anneal")
        print(f"    B  {tm_note}  {pb.sequence}  (antisense / reverse)")

    if result.primer_C:
        pc = result.primer_C
        tm_note = (f"Tm={pc.tm:.0f}°C full / "
                   f"{pc.tm_anneal:.0f}°C anneal")
        print(f"    C  {tm_note}  {pc.sequence}  (sense / forward)")

    if result.primer_D:
        pd = result.primer_D
        if pd.enzyme is None:
            _prow("D", pd.sequence, pd.tm, "[user-supplied]")
        else:
            neb = NEB_CATALOG.get(pd.enzyme or "", {})
            neb_tag = f"NEB {neb['hf'] or neb['cat']} (HF)" if neb.get("hf") else \
                      (f"NEB {neb['cat']}" if neb else "")
            warn_tag = f"  ⚠ {neb['note']}" if neb.get("note") else ""
            _prow("D", pd.sequence, pd.tm,
                  f"[{pd.enzyme}  {neb_tag}  cut {pd.cut_pos}]{warn_tag}")
    else:
        print("    D  (not found)")

    # Overlap
    print(f"\n    Overlap  Tm={result.overlap_tm:.0f}°C  "
          f"{result.overlap_seq}  ({len(result.overlap_seq)} bp)")

    # Cut sites on the whole construct (not just within each small fragment) —
    # gives the full-sequence context of where every enzyme relevant to this
    # design actually cuts: primer A's site, primer D's site, AND the
    # diagnostic enzyme's site (the one you'd actually digest with to
    # confirm the clone — easy to forget since it's not "A" or "D").
    whole_cuts = _collect_all_cut_sites(result)
    if result.mutated_sequence and whole_cuts:
        print(f"\n  {_hr()}")
        print(f"  Cut sites in the full construct  "
              f"({len(result.mutated_sequence)} bp)")
        _print_seq_with_cuts(result.mutated_sequence, 0, whole_cuts, unit="full construct")

    # Full restriction digest map — every enzyme's cut site anywhere in the
    # construct, not just primer A/D/diagnostic. Shown for BOTH wild type and
    # mutant so the two can be compared directly, not just their diff.
    if result.mutated_sequence:
        mut_map = scan_sites(result.mutated_sequence)
        mut_cuts = [
            (pos, enz) for enz, positions in mut_map.items() for pos in positions
        ]
        if mut_cuts:
            print(f"\n  {_hr()}")
            print(f"  Full restriction digest map — mutant  "
                  f"({len(mut_cuts)} sites, {len(mut_map)} enzymes, "
                  f"{len(result.mutated_sequence)} bp)")
            _print_seq_with_cuts(result.mutated_sequence, 0, mut_cuts, unit="full construct")

    if result.original_sequence:
        wt_map = scan_sites(result.original_sequence)
        wt_cuts = [
            (pos, enz) for enz, positions in wt_map.items() for pos in positions
        ]
        if wt_cuts:
            print(f"\n  {_hr()}")
            print(f"  Full restriction digest map — wild type  "
                  f"({len(wt_cuts)} sites, {len(wt_map)} enzymes, "
                  f"{len(result.original_sequence)} bp)")
            _print_seq_with_cuts(result.original_sequence, 0, wt_cuts, unit="wild-type construct")

    # PCR products (sizes + sequences, with cut sites marked)
    if result.frag_ad_bp:
        print(f"\n  {_hr()}")
        print(f"  Predicted PCR products")

        pa, pd = result.primer_A, result.primer_D
        print(f"    Fragment ab  {result.frag_ab_bp:>7} bp  "
              f"(primer A → overlap end)")
        _print_seq_with_cuts(
            result.frag_ab_seq, pa.start if pa else 0,
            [(pa.cut_pos, pa.enzyme) if pa else (None, None)],
        )
        print(f"    Fragment cd  {result.frag_cd_bp:>7} bp  "
              f"(overlap start → primer D)")
        cd_start = result.bc_result.overlap_start if result.bc_result else 0
        _print_seq_with_cuts(
            result.frag_cd_seq, cd_start,
            [(pd.cut_pos, pd.enzyme) if pd else (None, None)],
        )
        print(f"    Assembled    {result.frag_ad_bp:>7} bp")
        _print_seq_with_cuts(
            result.frag_ad_seq, pa.start if pa else 0,
            [
                (pa.cut_pos, pa.enzyme) if pa else (None, None),
                (pd.cut_pos, pd.enzyme) if pd else (None, None),
            ],
        )

    # All A/D candidates if requested
    if show_all_candidates and result.ad_result_a and result.ad_result_d:
        print(f"\n  {_hr()}")
        print("  All primer A candidates")
        _print_candidates("A", result.ad_result_a, show_all=True)
        print()
        print("  All primer D candidates")
        _print_candidates("D", result.ad_result_d, show_all=True)

    # Verification
    print(f"\n  {_hr()}")
    print("  Verification")

    def _vline(label, passed, detail):
        icon = "✓ PASS" if passed else ("✗ FAIL" if passed is False else "— SKIP")
        print(f"    {label:<26} {icon}")
        if passed is False or (passed is None and detail):
            print(f"      {detail}")

    _vline("Translation check", result.translation_passed, result.translation_detail)
    _vline("Restriction site check", result.restriction_passed, result.restriction_detail)

    print(f"\n  {_hr('═')}")
    overall_icon = "✓ PASS" if result.overall_passed else \
                   ("— INCOMPLETE" if result.overall_passed is None else "✗ FAIL")
    print(f"  Overall                        {overall_icon}")
    print(f"  {_hr('═')}")
    print()


# ---------------------------------------------------------------------------
# ORF start detection / prompting
# ---------------------------------------------------------------------------

def _find_all_orfs(
    sequence: str,
    top_n: int = 5,
    min_aa: int = 20,
) -> list[tuple[int, int]]:
    """
    Return up to top_n (0-based nt start, translated aa length) ORF candidates,
    sorted by translated length descending.

    Every ATG in the sequence is a *candidate*, but most are spurious —
    in-frame with a stop codon a few codons later, or too short to ever be a
    usable ORF. Rather than presenting those as if they were real options,
    each candidate is checked here first (does it actually translate, and is
    the protein at least min_aa residues before it fails/exits) and only
    ones that pass are returned.
    """
    from Bio.Seq import Seq as _Seq
    orfs = []
    pos = 0
    while True:
        atg = sequence.find("ATG", pos)
        if atg == -1:
            break
        try:
            prot = str(_Seq(sequence[atg:]).translate())
        except Exception:
            pos = atg + 1
            continue
        stop = prot.find("*")
        length = stop if stop != -1 else len(prot)
        if length >= min_aa:
            orfs.append((atg, length))
        pos = atg + 1
    orfs.sort(key=lambda t: -t[1])
    return orfs[:top_n]


def _resolve_orf_start(sequence: str, args: argparse.Namespace) -> int:
    """
    Resolve the 0-based ORF start to use for the whole run.

    If --orf-start was given, use it directly. Otherwise auto-detect the
    longest ORF and, on an interactive terminal, ask the user to confirm or
    override before doing anything else. On a non-interactive stream (piped
    input, script, CI), fall back silently to the auto-detected value.
    """
    if args.orf_start is not None:
        return args.orf_start

    candidates = _find_all_orfs(sequence, top_n=5, min_aa=20)
    if not candidates:
        # Every ATG failed the min_aa=20 sanity check — fall back to any ATG
        # at all rather than dead-ending, but say so plainly.
        candidates = _find_all_orfs(sequence, top_n=5, min_aa=1)
        if not candidates:
            sys.exit("ERROR: No ATG start codon found anywhere in the sequence. "
                      "Supply --orf-start manually if you know the ORF position.")
        print("WARNING: No ATG produces a protein ≥20 aa — the sequence may not "
              "contain a real ORF, or is truncated/wrong-frame. Showing the "
              "longest available ATG(s) anyway.", file=sys.stderr)

    best_start, best_len = candidates[0]

    if not sys.stdin.isatty():
        print(f"(Using longest ORF, starting at nt {best_start} ({best_len} aa). "
              f"Add --orf-start N to choose a different one.)", file=sys.stderr)
        return best_start

    print(f"\nORF start not specified (--orf-start).", file=sys.stderr)
    print(f"Found {len(candidates)} candidate start codon(s) "
          f"(longest translated ORF first):", file=sys.stderr)
    for i, (start, length) in enumerate(candidates, 1):
        marker = "  ◀ longest" if i == 1 else ""
        print(f"  {i}. nt {start}  ({length} aa){marker}", file=sys.stderr)

    prompt = (
        f"\nPick one — type just the list number (e.g. \"1\"), "
        f"or press Enter for the longest ORF (option 1, nt {best_start}): "
    )
    while True:
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return best_start

        if raw == "":
            return best_start
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(candidates):
                return candidates[n - 1][0]
            print(f"  '{n}' isn't one of the list numbers 1-{len(candidates)}. "
                  f"Try again, or press Enter for the default.", file=sys.stderr)
            continue
        print(f"  Didn't understand '{raw}' — please type just a single number "
              f"from 1-{len(candidates)}, or press Enter for the default.",
              file=sys.stderr)


# ---------------------------------------------------------------------------
# Output tee (console + optional file)
# ---------------------------------------------------------------------------

class _Tee:
    """Write to both the real stdout and a file, transparently."""

    def __init__(self, real_stdout, file_handle):
        self._real = real_stdout
        self._file = file_handle

    def write(self, text):
        self._real.write(text)
        self._file.write(text)

    def flush(self):
        self._real.flush()
        self._file.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.neb_info:
        from primer_ad import print_neb_table
        print_neb_table()
        sys.exit(0)

    if args.fasta_file is None:
        parser.error("SEQUENCE_FILE is required (unless using --neb-info).")

    sequence = _read_sequence_file(args.fasta_file)

    # Resolve ORF start once, up front, for both --find and single-mutation
    # modes — prompting interactively if not supplied via --orf-start.
    args.orf_start = _resolve_orf_start(sequence, args)

    # ── --find mode: list positions where primer design will actually work ──
    if args.find:
        from pipeline import find_all_positions

        real_stdout = sys.stdout
        out_file = None
        if args.output:
            out_file = open(args.output, "w")
            sys.stdout = _Tee(real_stdout, out_file)

        orig_aa_f = args.find[0].upper()
        new_aa_f  = args.find[1].upper()
        orf = args.orf_start

        hits = find_all_positions(sequence, orf, orig_aa_f)

        if not hits:
            print(f"\nNo {orig_aa_f} residues found in the protein "
                  f"(ORF start nt {orf}).")
            sys.exit(1)

        print(f"\nChecking {len(hits)} candidate {orig_aa_f} position(s) "
              f"for feasible primer design…", file=sys.stderr)

        # Actually run the design for every candidate to filter out positions
        # where primer design would fail (too close to sequence ends, no
        # diagnostic site possible, etc).
        working: list[tuple] = []
        failing: list[tuple] = []
        for pos_1, codon in hits:
            r = design_mutation_primers(
                sequence, pos_1, orig_aa_f, new_aa_f,
                orf_start=orf,
                tm_range=(args.tm_min, args.tm_max),
                window_bp=tuple(args.window),
                min_overlap=args.min_overlap,
            )
            if r.errors:
                failing.append((pos_1, codon, r.errors[0]))
            elif r.primer_A is None or r.primer_D is None:
                reason = "no unique flanking restriction site for primer A or D"
                failing.append((pos_1, codon, reason))
            elif r.translation_passed is False:
                failing.append((pos_1, codon, r.translation_detail))
            elif r.restriction_passed is False:
                # Diagnostic site exists but verification on the assembled
                # product actually failed — this would show Overall ✗ FAIL
                # if run individually, so it must not be listed as working.
                failing.append((pos_1, codon, r.restriction_detail))
            elif r.overall_passed is not True:
                # Catch-all: anything that isn't a confirmed, verified PASS
                # (e.g. verification was skipped or another check failed)
                # is not offered as a working option.
                failing.append((pos_1, codon,
                                 "verification did not confirm an overall PASS"))
            else:
                if r.diagnostic is None:
                    note = " [no diagnostic site — translation-only check]"
                elif r.diagnostic.source == "silent_mutation":
                    note = (f" [needs silent aa{r.diagnostic.silent_aa_index} "
                            f"{r.diagnostic.silent_original_codon}→"
                            f"{r.diagnostic.silent_new_codon} for {r.diagnostic.enzyme} site]")
                else:
                    note = ""
                working.append((pos_1, codon, note))

        extended_search_used = False

        # ── If nothing worked on first pass, retry with wider silent-mutation
        #    window (up to ±500 nt) to find silent diagnostic site options ──
        if not working:
            print(
                "\nNo positions designable with default settings. "
                "Retrying with wider silent-mutation search (±500 nt)…",
                file=sys.stderr,
            )
            extended_search_used = True
            for pos_1, codon, fail_reason in list(failing):
                if "flanking restriction site" in fail_reason:
                    # A/D primer failure — wider silent search won't help; skip
                    continue
                r2 = design_mutation_primers(
                    sequence, pos_1, orig_aa_f, new_aa_f,
                    orf_start=orf,
                    tm_range=(args.tm_min, args.tm_max),
                    window_bp=tuple(args.window),
                    min_overlap=args.min_overlap,
                    max_silent_search_flank=500,
                )
                if (not r2.errors
                        and r2.primer_A is not None
                        and r2.primer_D is not None
                        and r2.overall_passed is True):
                    if r2.diagnostic and r2.diagnostic.source == "silent_mutation":
                        note = (
                            f" [needs silent aa{r2.diagnostic.silent_aa_index} "
                            f"{r2.diagnostic.silent_original_codon}→"
                            f"{r2.diagnostic.silent_new_codon} "
                            f"for {r2.diagnostic.enzyme} diagnostic site]"
                        )
                    else:
                        note = " [no diagnostic site — translation-only check]"
                    working.append((pos_1, codon, note))
                    failing = [(p, c, rr) for p, c, rr in failing if p != pos_1]

        # ── Print results ─────────────────────────────────────────────────────
        if working:
            label = (
                f"designable with extended silent-mutation search"
                if extended_search_used else
                f"designable"
            )
            print(f"\n{orig_aa_f}→{new_aa_f}: {len(working)} of {len(hits)} "
                  f"position(s) {label}:\n")
            print(f"  {'Position':<10} {'Codon':<8}  Command to run")
            print(f"  {'-'*10} {'-'*8}  {'-'*50}")
            for pos_1, codon, note in working:
                cmd = (f"python3 sdm_design.py {args.fasta_file} "
                       f"--orf-start {orf} "
                       f"--mutation {orig_aa_f}{pos_1}{new_aa_f}")
                print(f"  {orig_aa_f}{pos_1:<9} {codon:<8}  {cmd}{note}")
        else:
            print(
                f"\nNo {orig_aa_f}→{new_aa_f} positions are designable "
                f"even with extended silent-mutation search (±500 nt)."
            )

        if failing and not args.show_all:
            print(f"\n{len(failing)} position(s) filtered out "
                  f"(use --show-all to see why).")
        elif failing:
            print(f"\nFiltered out ({len(failing)}):")
            for pos_1, codon, reason in failing:
                print(f"  {orig_aa_f}{pos_1:<8} {codon:<6}  {reason[:90]}")

        sys.stdout = real_stdout
        if out_file is not None:
            out_file.close()

        if args.output:
            print(f"\nSaved to {args.output}", file=sys.stderr)
        else:
            print(
                f"\nTo save this output to a file, rerun with:\n"
                f"  python3 sdm_design.py {args.fasta_file} --orf-start {orf} "
                f"--find {orig_aa_f} {new_aa_f} --output {orig_aa_f}to{new_aa_f}_positions.txt",
                file=sys.stderr,
            )

        sys.exit(0 if working else 1)

    mutations = _resolve_mutations(args)
    label = " + ".join(f"{oa}{p}{na}" for oa, p, na in mutations)

    orf_note = f"ORF start {args.orf_start}" if args.orf_start is not None \
               else "ORF start auto-detect"
    print(
        f"Designing primers for {label} "
        f"in {args.fasta_file} ({len(sequence)} bp, {orf_note})…",
        file=sys.stderr,
    )

    result = design_mutation_primers(
        sequence=sequence,
        target_position=[p for _, p, _ in mutations],
        original_aa=[oa for oa, _, _ in mutations],
        new_aa=[na for _, _, na in mutations],
        orf_start=args.orf_start,
        tm_range=(args.tm_min, args.tm_max),
        window_bp=tuple(args.window),
        min_overlap=args.min_overlap,
        primer_A_seq=args.primer_a_seq,
        primer_D_seq=args.primer_d_seq,
    )

    real_stdout = sys.stdout
    out_file = None
    if args.output:
        out_file = open(args.output, "w")
        sys.stdout = _Tee(real_stdout, out_file)

    try:
        if args.json:
            print(json.dumps(result_to_dict(result), indent=2))
        else:
            _print_formatted(result, show_all_candidates=args.all_candidates or args.show_all)
    finally:
        sys.stdout = real_stdout
        if out_file is not None:
            out_file.close()

    if args.output:
        print(f"\nSaved to {args.output}", file=sys.stderr)
    else:
        save_cmd = " ".join(
            (["python3", "sdm_design.py", args.fasta_file, "--orf-start", str(args.orf_start)]
             if args.orf_start is not None else
             ["python3", "sdm_design.py", args.fasta_file])
            + (["--mutation", label.replace(" ", "")])
            + (["--json"] if args.json else [])
            + ["--output", f"{label.replace(' ', '')}_report.{'json' if args.json else 'txt'}"]
        )
        print(f"\nTo save this output to a file, rerun with:\n  {save_cmd}", file=sys.stderr)

    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
