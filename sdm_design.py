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
                        Alternatively supply --orig-aa, --position, --new-aa.
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

from pipeline import design_mutation_primers, parse_mutation_label, result_to_dict
from assembly import print_report
from primer_ad import NEB_CATALOG


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
    mut_group.add_argument("--mutation", metavar="e.g. K255E",
                           help="Compact mutation label.")
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


def _resolve_mutation(args: argparse.Namespace) -> tuple[str, int, str]:
    """Return (original_aa, position, new_aa) or exit with an error."""
    if args.mutation:
        try:
            return parse_mutation_label(args.mutation)
        except ValueError as exc:
            sys.exit(f"ERROR: {exc}")

    if args.orig_aa and args.position and args.new_aa:
        return args.orig_aa.upper(), args.position, args.new_aa.upper()

    sys.exit(
        "ERROR: Provide either --mutation (e.g. K255E) "
        "or all three of --orig-aa, --position, --new-aa."
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

    # Primers
    print(f"\n  {_hr()}")
    print(f"  Primers  (5'→3')")

    def _prow(lbl, seq, tm, note=""):
        tag = f"Tm={tm:.0f}°C"
        print(f"    {lbl}  {tag:<10}  {seq}  {note}")

    if result.primer_A:
        pa = result.primer_A
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
        _prow("B", pb.sequence, pb.tm, "(sense / forward)")

    if result.primer_C:
        pc = result.primer_C
        tm_note = (f"Tm={pc.tm:.0f}°C full / "
                   f"{pc.tm_anneal:.0f}°C anneal")
        print(f"    C  {tm_note}  {pc.sequence}  (antisense / reverse)")

    if result.primer_D:
        pd = result.primer_D
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

    # Fragment sizes
    if result.frag_ad_bp:
        print(f"\n  {_hr()}")
        print(f"  Predicted PCR fragments")
        print(f"    Fragment ab  {result.frag_ab_bp:>7} bp  "
              f"(primer A → overlap end)")
        print(f"    Fragment cd  {result.frag_cd_bp:>7} bp  "
              f"(overlap start → primer D)")
        print(f"    Assembled    {result.frag_ad_bp:>7} bp")

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

    orig_aa, position, new_aa = _resolve_mutation(args)

    orf_note = f"ORF start {args.orf_start}" if args.orf_start is not None \
               else "ORF start auto-detect"
    print(
        f"Designing primers for {orig_aa}{position}{new_aa} "
        f"in {args.fasta_file} ({len(sequence)} bp, {orf_note})…",
        file=sys.stderr,
    )

    result = design_mutation_primers(
        sequence=sequence,
        target_position=position,
        original_aa=orig_aa,
        new_aa=new_aa,
        orf_start=args.orf_start,
        tm_range=(args.tm_min, args.tm_max),
        window_bp=tuple(args.window),
        min_overlap=args.min_overlap,
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
            + (["--mutation", f"{orig_aa}{position}{new_aa}"])
            + (["--json"] if args.json else [])
            + ["--output", f"{orig_aa}{position}{new_aa}_report.{'json' if args.json else 'txt'}"]
        )
        print(f"\nTo save this output to a file, rerun with:\n  {save_cmd}", file=sys.stderr)

    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
