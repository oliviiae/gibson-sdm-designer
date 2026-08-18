"""
Web app for Gibson primer design.

Run locally:
    streamlit run app.py

This is a thin UI layer only — all the actual design logic lives in
pipeline.py / primer_bc.py / primer_ad.py / assembly.py / restriction_utils.py
/ codon_utils.py, unchanged. The app just calls those same functions.
"""

import io
import tempfile

import streamlit as st
from Bio import SeqIO
from Bio.Seq import Seq

from pipeline import (
    design_mutation_primers, find_all_positions, parse_mutation_label, parse_mutation_labels,
)
from primer_ad import NEB_CATALOG
from assembly import translate_orf
from xdna_utils import read_xdna_sequence
from restriction_utils import scan_sites

st.set_page_config(page_title="Gibson Primer Designer", layout="wide")


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  html, body, [class*="css"] {
    font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  }
  #MainMenu, footer, header { visibility: hidden; }

  .app-title {
    font-size: 1.65rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #1a2530;
    margin-bottom: 0.15rem;
  }
  .app-subtitle {
    font-size: 0.92rem;
    color: #667380;
    margin-bottom: 1.6rem;
  }
  .section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #8a97a3;
    margin: 1.1rem 0 0.35rem 0;
  }
  .result-card {
    border: 1px solid #e3e7eb;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
    background: #fbfcfd;
  }
  .kv-row {
    display: flex;
    justify-content: space-between;
    padding: 0.28rem 0;
    border-bottom: 1px solid #eef1f3;
    font-size: 0.88rem;
  }
  .kv-row:last-child { border-bottom: none; }
  .kv-label { color: #667380; }
  .kv-value { color: #1a2530; font-weight: 500; font-variant-numeric: tabular-nums; }
  .kv-value code { background: #eef1f3; padding: 0.1rem 0.35rem; border-radius: 4px; }

  .badge {
    display: inline-block;
    padding: 0.18rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
  }
  .badge-pass { background: #e4f3ea; color: #1e6b3c; }
  .badge-fail { background: #fbe7e7; color: #a33232; }
  .badge-skip { background: #f0f1f3; color: #6b7480; }

  .verify-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid #eef1f3;
  }
  .verify-row:last-child { border-bottom: none; }
  .verify-label { font-size: 0.88rem; color: #3a444d; }

  div.stButton > button, div.stDownloadButton > button {
    background: #1a2530;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 500;
  }
  div.stButton > button:hover, div.stDownloadButton > button:hover {
    background: #2c3946;
    color: white;
  }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_fasta_text(text: str) -> str | None:
    try:
        records = list(SeqIO.parse(io.StringIO(text), "fasta"))
    except Exception as exc:
        st.error(f"Could not parse FASTA: {exc}")
        return None
    if not records:
        st.error("No sequences found in the pasted/uploaded FASTA.")
        return None
    if len(records) > 1:
        st.warning(f"{len(records)} records found; using the first "
                   f"({records[0].id}, {len(records[0])} bp).")
    seq = str(records[0].seq).upper()
    if not seq:
        st.error("First FASTA record is empty.")
        return None
    return seq


def _read_xdna_upload(upload) -> str | None:
    """
    Extract a sequence from an uploaded .xdna file. This format has no
    public spec — read_xdna_sequence heuristically finds the longest
    DNA-like byte run, so the result is shown for the user to sanity-check
    (same caveat the CLI's --xdna path prints).
    """
    with tempfile.NamedTemporaryFile(suffix=".xdna", delete=False) as tmp:
        tmp.write(upload.getvalue())
        tmp_path = tmp.name
    try:
        result = read_xdna_sequence(tmp_path)
    except ValueError as exc:
        st.error(
            f"Could not read .xdna file: {exc}\n\n"
            "Safer alternative: open the file in SnapGene / Serial Cloner / "
            "ApE and export it as FASTA instead."
        )
        return None
    st.warning(
        f".xdna has no public file spec — please double-check this against your "
        f"sequence viewer before ordering primers.\n\n"
        f"Extracted {result.length} bp from a {result.file_size}-byte file"
        + (f" ({result.candidate_runs_found} DNA-like runs found; used the longest)."
           if result.candidate_runs_found > 1 else ".")
        + f"\n\nFirst 60bp: `{result.preview_start}`\n\nLast 60bp: `{result.preview_end}`"
    )
    return result.sequence


def _find_orf_candidates(sequence: str, top_n: int = 8, min_aa: int = 20):
    orfs = []
    pos = 0
    while True:
        atg = sequence.find("ATG", pos)
        if atg == -1:
            break
        try:
            prot = str(Seq(sequence[atg:]).translate())
        except Exception:
            pos = atg + 1
            continue
        stop = prot.find("*")
        length = stop if stop != -1 else len(prot)
        if length >= min_aa:
            orfs.append((atg, length))
        pos = atg + 1
    orfs.sort(key=lambda t: -t[1])
    if not orfs:
        pos = 0
        while True:
            atg = sequence.find("ATG", pos)
            if atg == -1:
                break
            try:
                prot = str(Seq(sequence[atg:]).translate())
            except Exception:
                pos = atg + 1
                continue
            stop = prot.find("*")
            length = stop if stop != -1 else len(prot)
            orfs.append((atg, length))
            pos = atg + 1
        orfs.sort(key=lambda t: -t[1])
    return orfs[:top_n]


def _neb_tag(enzyme: str | None) -> str:
    if not enzyme:
        return ""
    info = NEB_CATALOG.get(enzyme)
    if not info:
        return ""
    tag = f"NEB {info['hf']} (HF)" if info.get("hf") else f"NEB {info['cat']}"
    if info.get("note"):
        tag += f" — {info['note']}"
    return tag


def _badge(passed) -> str:
    if passed is True:
        return '<span class="badge badge-pass">PASS</span>'
    if passed is False:
        return '<span class="badge badge-fail">FAIL</span>'
    return '<span class="badge badge-skip">SKIPPED</span>'


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown('<div class="app-title">Gibson Primer Designer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">'
    'Automated primer design for Gibson-assembly site-directed mutagenesis, '
    'with restriction-site verification and NEB catalog lookup.'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar: sequence input
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="section-label">Sequence</div>', unsafe_allow_html=True)
    upload = st.file_uploader("Sequence file", type=["fasta", "fa", "txt", "xdna"],
                               label_visibility="collapsed")
    pasted = st.text_area("Or paste FASTA text", height=140,
                           placeholder=">construct\nATGGCT...", label_visibility="collapsed")

    sequence = None
    if upload is not None:
        if upload.name.lower().endswith(".xdna"):
            sequence = _read_xdna_upload(upload)
        else:
            sequence = _read_fasta_text(upload.getvalue().decode("utf-8", errors="replace"))
    elif pasted.strip():
        sequence = _read_fasta_text(pasted)

    if sequence:
        st.caption(f"{len(sequence):,} bp loaded")

    st.markdown('<div class="section-label">Reading frame</div>', unsafe_allow_html=True)
    orf_start = None
    if sequence:
        candidates = _find_orf_candidates(sequence)
        if candidates:
            labels = [f"nt {start}  ·  {length} aa{'  (longest)' if i == 0 else ''}"
                      for i, (start, length) in enumerate(candidates)]
            labels.append("Custom position…")
            choice = st.selectbox("ORF start codon (ATG)", labels, index=0, label_visibility="collapsed")
            if choice == "Custom position…":
                orf_start = st.number_input("Custom ORF start (0-based nt)",
                                             min_value=0, max_value=len(sequence) - 1, value=0)
            else:
                orf_start = candidates[labels.index(choice)][0]
        else:
            st.error("No ATG found in this sequence.")
    else:
        st.caption("Load a sequence first")

    st.markdown('<div class="section-label">Parameters</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    tm_min = c1.number_input("Tm min (°C)", value=48.0, step=1.0)
    tm_max = c2.number_input("Tm max (°C)", value=54.0, step=1.0)
    c3, c4 = st.columns(2)
    win_near = c3.number_input("Window near (bp)", value=350, step=50)
    win_far = c4.number_input("Window far (bp)", value=500, step=50)
    st.caption("A/D flanking search, applied per side")


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

if not sequence or orf_start is None:
    st.info("Load a FASTA sequence in the sidebar to begin.")
    st.stop()

mode = st.radio("Mode", ["Design a mutation", "Scan for designable positions"],
                 horizontal=True, label_visibility="collapsed")
st.write("")


def _bracket_spans(seq: str, spans: list[tuple[int, int]]) -> str:
    """Wrap each [start, end) span in seq with [ ]. spans may be unsorted/adjacent."""
    out = seq
    for start, end in sorted(spans, key=lambda s: -s[0]):
        out = out[:start] + "[" + out[start:end] + "]" + out[end:]
    return out


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


def _render_seq_with_cuts(
    seq: str, frag_start: int, cuts: list[tuple[int | None, str | None]],
    unit: str = "this fragment",
):
    """
    Render seq (monospace) with each restriction enzyme's cut site
    highlighted in place. cuts: list of (cut_pos_1based_in_full_construct,
    enzyme_name) — mirrors the CLI's cut-site marking exactly. frag_start is
    the 0-based absolute position where seq begins in the full construct
    (0 when seq itself IS the full construct).
    """
    marked = seq
    labels = []
    for cut_pos, enzyme in cuts:
        if cut_pos is None:
            continue
        rel = (cut_pos - 1) - frag_start
        if 0 <= rel < len(seq):
            labels.append((rel, enzyme))
    # Multiple enzymes can cut at the same position — wrap each unique
    # position only once (in descending order so earlier insertions don't
    # shift later indices). Wrapping the same index twice would splice into
    # the already-inserted <span> tag itself, corrupting the HTML.
    for rel in sorted({r for r, _ in labels}, reverse=True):
        marked = (
            marked[:rel]
            + f'<span style="background:#fde8e8;color:#a33232;font-weight:700;">{marked[rel]}</span>'
            + marked[rel + 1:]
        )
    st.markdown(
        f'<div style="font-family:monospace;font-size:0.85rem;word-break:break-all;'
        f'background:#F5F8F8;padding:0.6rem;border-radius:6px;">{marked}</div>',
        unsafe_allow_html=True,
    )
    if labels:
        st.caption(
            "Cut site(s): " + ", ".join(
                f"{enzyme} at nt {rel} in {unit}" for rel, enzyme in sorted(labels)
            )
        )


def _mark_dna_mutations(
    dna_window: str,
    win_start: int,
    positions: list[int],
    original_codons: list[str],
    new_codons: list[str],
) -> str:
    """
    Color-code dna_window: the rest of a changed codon is shown in red/bold
    for context, and the specific nucleotide(s) that actually differ from
    wild type within that codon get a distinct highlighted background —
    since a single point mutation usually changes only 1 of the 3 bases in
    its codon, not the whole thing.
    """
    n = len(dna_window)
    styles = [None] * n  # None / "codon" / "nt"
    for i, pos in enumerate(positions):
        codon_start = (pos - win_start) * 3
        if codon_start < 0 or codon_start + 3 > n:
            continue
        orig = original_codons[i] if i < len(original_codons) else ""
        new = new_codons[i] if i < len(new_codons) else ""
        for j in range(3):
            idx = codon_start + j
            if styles[idx] is None:
                styles[idx] = "codon"
            if j < len(orig) and j < len(new) and orig[j] != new[j]:
                styles[idx] = "nt"

    out = []
    for ch, style in zip(dna_window, styles):
        if style == "nt":
            out.append(
                '<span style="background:#ffd43b;color:#7a4a00;'
                'font-weight:800;border-radius:3px;padding:0 1px;">'
                f"{ch}</span>"
            )
        elif style == "codon":
            out.append(f'<span style="color:#c0392b;font-weight:700;">{ch}</span>')
        else:
            out.append(ch)
    return "".join(out)


def _render_mutated_region(result, aa_flank: int = 15):
    """Show the mutated ORF's DNA + protein for a window around the
    mutation(s), with the changed codon(s)/residue(s) bracketed."""
    if not result.mutated_sequence or not result.mutation_positions:
        return

    orf_start = result.orf_start_detected
    mutated_seq = result.mutated_sequence
    positions = result.mutation_positions

    try:
        protein = translate_orf(mutated_seq, orf_start)
    except ValueError:
        return
    if not protein:
        return

    aa_min, aa_max = min(positions), max(positions)
    win_start = max(1, aa_min - aa_flank)
    win_end = min(len(protein), aa_max + aa_flank)

    dna_start = orf_start + (win_start - 1) * 3
    dna_end = orf_start + win_end * 3
    dna_window = mutated_seq[dna_start:dna_end]
    protein_window = protein[win_start - 1: win_end]

    residue_spans = [
        (pos - win_start, pos - win_start + 1)
        for pos in positions if win_start <= pos <= win_end
    ]

    dna_marked = _mark_dna_mutations(
        dna_window, win_start, positions,
        result.original_codons, result.new_codons,
    )
    protein_marked = _bracket_spans(protein_window, residue_spans).replace(
        "[", '<span style="color:#c0392b;font-weight:700;">['
    ).replace("]", ']</span>')

    st.markdown(
        f'<div class="section-label">Mutated region (aa {win_start}-{win_end})</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="result-card">'
        f'<div class="kv-row"><span class="kv-label">DNA</span>'
        f'<span class="kv-value" style="font-family:monospace;word-break:break-all;">'
        f"5'-{dna_marked}-3'</span></div>"
        f'<div class="kv-row"><span class="kv-label">Protein</span>'
        f'<span class="kv-value" style="font-family:monospace;word-break:break-all;">'
        f'{protein_marked}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Red = changed codon/residue · yellow highlight = the exact "
        "nucleotide(s) that differ from wild type"
    )


def _render_cut_site_diff(result):
    """
    Every NEB enzyme gained/lost anywhere in the construct, wild type vs
    mutant — not just the one enzyme picked as the diagnostic, so the full
    restriction-map impact of the edit is visible at a glance.
    """
    diff = getattr(result, "cut_site_diff", None) or {}
    gained = diff.get("gained", {})
    lost = diff.get("lost", {})
    if not gained and not lost:
        return

    st.markdown(
        '<div class="section-label">Cutting pattern diff (wild type vs mutant)</div>',
        unsafe_allow_html=True,
    )
    rows = []
    for enz in sorted(gained):
        positions = ", ".join(str(p) for p in gained[enz])
        rows.append(["+ gained", enz, positions, "mutant only"])
    for enz in sorted(lost):
        positions = ", ".join(str(p) for p in lost[enz])
        rows.append(["− lost", enz, positions, "wild type only"])
    st.dataframe(
        {"Change": [r[0] for r in rows],
         "Enzyme": [r[1] for r in rows],
         "Position(s)": [r[2] for r in rows],
         "Present in": [r[3] for r in rows]},
        hide_index=True,
        use_container_width=True,
    )


def _render_result(result, key_prefix=""):
    if result.errors:
        for e in result.errors:
            st.error(e)
        return

    kv = [
        ("Mutation", result.mutation_label),
        ("Codon change", f"<code>{result.original_codon}</code> → <code>{result.new_codon}</code>"),
        ("ORF start", f"nt {result.orf_start_detected} (0-based)"),
        ("Mutation site", f"nt {result.mutation_nt_position} (0-based)"),
    ]
    if result.diagnostic:
        d = result.diagnostic
        src = "from mutation" if d.source == "mutation" else \
              (f"silent mutation at aa{d.silent_aa_index}, "
               f"{d.silent_original_codon}→{d.silent_new_codon}")
        kv.append(("Diagnostic site", f"{d.enzyme} — {d.effect} ({src})"))
    else:
        kv.append(("Diagnostic site", "none found"))
    rows_html = "".join(
        f'<div class="kv-row"><span class="kv-label">{label}</span>'
        f'<span class="kv-value">{value}</span></div>'
        for label, value in kv
    )
    st.markdown(f'<div class="result-card">{rows_html}</div>', unsafe_allow_html=True)

    _render_mutated_region(result)
    _render_cut_site_diff(result)

    st.markdown('<div class="section-label">Primers (5\' → 3\')</div>', unsafe_allow_html=True)
    rows = []
    if result.primer_A:
        pa = result.primer_A
        note = "user-supplied" if pa.enzyme is None else \
               f"{pa.enzyme}  ·  {_neb_tag(pa.enzyme)}  ·  cut {pa.cut_pos}"
        rows.append(["A", pa.sequence, f"{pa.tm:.0f}°C", note])
    if result.primer_B:
        pb = result.primer_B
        rows.append(["B", pb.sequence, f"{pb.tm:.0f}°C", "sense (forward)"])
    if result.primer_C:
        pc = result.primer_C
        rows.append(["C", pc.sequence,
                     f"{pc.tm:.0f}°C full / {pc.tm_anneal:.0f}°C anneal",
                     "antisense (reverse)"])
    if result.primer_D:
        pd = result.primer_D
        note = "user-supplied" if pd.enzyme is None else \
               f"{pd.enzyme}  ·  {_neb_tag(pd.enzyme)}  ·  cut {pd.cut_pos}"
        rows.append(["D", pd.sequence, f"{pd.tm:.0f}°C", note])
    st.dataframe(
        {"Primer": [r[0] for r in rows],
         "Sequence": [r[1] for r in rows],
         "Tm": [r[2] for r in rows],
         "Notes": [r[3] for r in rows]},
        hide_index=True,
        use_container_width=True,
    )

    st.markdown('<div class="section-label">Overlap &amp; fragments</div>', unsafe_allow_html=True)
    frag_html = (
        f'<div class="kv-row"><span class="kv-label">Overlap sequence</span>'
        f'<span class="kv-value"><code>{result.overlap_seq}</code> '
        f'({len(result.overlap_seq)} bp, Tm={result.overlap_tm:.0f}°C)</span></div>'
    )
    if result.frag_ad_bp:
        frag_html += (
            f'<div class="kv-row"><span class="kv-label">Fragment sizes</span>'
            f'<span class="kv-value">ab {result.frag_ab_bp} bp  ·  '
            f'cd {result.frag_cd_bp} bp  ·  assembled {result.frag_ad_bp} bp</span></div>'
        )
    st.markdown(f'<div class="result-card">{frag_html}</div>', unsafe_allow_html=True)

    whole_cuts = _collect_all_cut_sites(result)
    if result.mutated_sequence and whole_cuts:
        with st.expander(
            f"Show cut sites in the full construct ({len(result.mutated_sequence)} bp)"
        ):
            _render_seq_with_cuts(result.mutated_sequence, 0, whole_cuts, unit="the full construct")

    if result.mutated_sequence:
        mut_map = scan_sites(result.mutated_sequence)
        mut_cuts = [
            (pos, enz) for enz, positions in mut_map.items() for pos in positions
        ]
        if mut_cuts:
            with st.expander(
                f"Show full restriction digest map — mutant ({len(mut_cuts)} sites, "
                f"{len(mut_map)} enzymes)"
            ):
                _render_seq_with_cuts(result.mutated_sequence, 0, mut_cuts, unit="the mutant construct")

    if result.original_sequence:
        wt_map = scan_sites(result.original_sequence)
        wt_cuts = [
            (pos, enz) for enz, positions in wt_map.items() for pos in positions
        ]
        if wt_cuts:
            with st.expander(
                f"Show full restriction digest map — wild type ({len(wt_cuts)} sites, "
                f"{len(wt_map)} enzymes)"
            ):
                _render_seq_with_cuts(result.original_sequence, 0, wt_cuts, unit="the wild-type construct")

    if result.frag_ad_bp:
        with st.expander("Show PCR product sequences (cut sites marked)"):
            pa, pd = result.primer_A, result.primer_D
            cd_start = result.bc_result.overlap_start if result.bc_result else 0

            st.caption(f"Fragment ab — {result.frag_ab_bp} bp (primer A → overlap end)")
            _render_seq_with_cuts(
                result.frag_ab_seq, pa.start if pa else 0,
                [(pa.cut_pos, pa.enzyme)] if pa else [],
            )
            st.caption(f"Fragment cd — {result.frag_cd_bp} bp (overlap start → primer D)")
            _render_seq_with_cuts(
                result.frag_cd_seq, cd_start,
                [(pd.cut_pos, pd.enzyme)] if pd else [],
            )
            st.caption(f"Assembled product — {result.frag_ad_bp} bp")
            assembled_cuts = []
            if pa:
                assembled_cuts.append((pa.cut_pos, pa.enzyme))
            if pd:
                assembled_cuts.append((pd.cut_pos, pd.enzyme))
            _render_seq_with_cuts(result.frag_ad_seq, pa.start if pa else 0, assembled_cuts)

    st.markdown('<div class="section-label">Verification</div>', unsafe_allow_html=True)
    verify_html = (
        f'<div class="verify-row"><span class="verify-label">Translation check</span>'
        f'{_badge(result.translation_passed)}</div>'
        f'<div class="verify-row"><span class="verify-label">Restriction site check</span>'
        f'{_badge(result.restriction_passed)}</div>'
        f'<div class="verify-row"><span class="verify-label"><strong>Overall</strong></span>'
        f'{_badge(result.overall_passed)}</div>'
    )
    st.markdown(f'<div class="result-card">{verify_html}</div>', unsafe_allow_html=True)

    report_text = (
        f"Mutation: {result.mutation_label}\n"
        f"Codon: {result.original_codon} -> {result.new_codon}\n"
        f"ORF start: {result.orf_start_detected}\n\n"
        + "\n".join(f"{r[0]}: {r[1]}  Tm={r[2]}  {r[3]}" for r in rows)
        + f"\n\nOverlap: {result.overlap_seq}\n"
        f"Overall: {'PASS' if result.overall_passed else 'FAIL'}\n"
    )
    if result.frag_ad_bp:
        report_text += (
            f"\nFragment ab ({result.frag_ab_bp} bp): {result.frag_ab_seq}\n"
            f"Fragment cd ({result.frag_cd_bp} bp): {result.frag_cd_seq}\n"
            f"Assembled product ({result.frag_ad_bp} bp): {result.frag_ad_seq}\n"
        )
    st.download_button("Download report", report_text,
                        file_name=f"{result.mutation_label}_report.txt",
                        key=f"{key_prefix}dl")


if mode == "Design a mutation":
    label = st.text_input(
        "Mutation", placeholder="e.g. K255E, or K255E+R300A for a combined double mutation",
        label_visibility="collapsed",
    )
    st.caption("Join two mutations with \"+\" to design one primer set for both "
               "(only realistic when the positions are close together).")

    with st.expander("Use your own primer A/D sequences (optional)"):
        col_a, col_d = st.columns(2)
        primer_a_input = col_a.text_input("Primer A sequence", key="primer_a_input")
        primer_d_input = col_d.text_input("Primer D sequence", key="primer_d_input")

    if st.button("Design primers", type="primary") and label:
        try:
            mutations = parse_mutation_labels(label)
        except ValueError as exc:
            st.error(str(exc))
        else:
            with st.spinner("Designing primers…"):
                result = design_mutation_primers(
                    sequence=sequence,
                    target_position=[p for _, p, _ in mutations],
                    original_aa=[oa for oa, _, _ in mutations],
                    new_aa=[na for _, _, na in mutations],
                    orf_start=int(orf_start),
                    tm_range=(tm_min, tm_max),
                    window_bp=(int(win_near), int(win_far)),
                    primer_A_seq=primer_a_input.strip() or None,
                    primer_D_seq=primer_d_input.strip() or None,
                )
            _render_result(result)

else:
    col1, col2, col3 = st.columns([1, 1, 3])
    from_aa = col1.text_input("From", value="K", max_chars=1).upper()
    to_aa = col2.text_input("To", value="E", max_chars=1).upper()

    # Streamlit buttons only return True on the single rerun right after
    # they're clicked. The scan results (and the "show design" button below)
    # need to survive later reruns triggered by the selectbox/button
    # interactions that follow, so they're persisted in session_state rather
    # than being recomputed and re-gated behind `if st.button("Scan sequence")`.
    if st.button("Scan sequence", type="primary") and from_aa and to_aa:
        hits = find_all_positions(sequence, int(orf_start), from_aa)
        if not hits:
            st.session_state["scan_results"] = None
            st.warning(f"No {from_aa} residues found in this reading frame.")
        else:
            progress = st.progress(0, text=f"Checking {len(hits)} candidate position(s)…")
            working, failing = [], []
            for i, (pos_1, codon) in enumerate(hits):
                r = design_mutation_primers(
                    sequence, pos_1, from_aa, to_aa,
                    orf_start=int(orf_start),
                    tm_range=(tm_min, tm_max),
                    window_bp=(int(win_near), int(win_far)),
                )
                if r.errors or r.primer_A is None or r.primer_D is None or r.overall_passed is not True:
                    failing.append((pos_1, codon))
                else:
                    note = ""
                    if r.diagnostic and r.diagnostic.source == "silent_mutation":
                        note = (f"needs silent aa{r.diagnostic.silent_aa_index} "
                                f"{r.diagnostic.silent_original_codon}→"
                                f"{r.diagnostic.silent_new_codon}")
                    working.append((pos_1, codon, note))
                progress.progress((i + 1) / len(hits))
            progress.empty()
            st.session_state["scan_results"] = {
                "from_aa": from_aa, "to_aa": to_aa,
                "hits": len(hits), "working": working, "failing": failing,
            }

    scan = st.session_state.get("scan_results")
    if scan and scan["from_aa"] == from_aa and scan["to_aa"] == to_aa:
        working, failing = scan["working"], scan["failing"]
        st.markdown(
            f'<div class="section-label">{len(working)} of {scan["hits"]} '
            f'position(s) designable</div>', unsafe_allow_html=True,
        )
        if working:
            st.dataframe(
                {"Position": [f"{from_aa}{p}{to_aa}" for p, _, _ in working],
                 "Codon": [c for _, c, _ in working],
                 "Note": [n for _, _, n in working]},
                hide_index=True,
                use_container_width=True,
            )
            picked = st.selectbox(
                "View full design for a position",
                [f"{from_aa}{p}{to_aa}" for p, _, _ in working],
            )
            if picked and st.button("Show design"):
                orig_aa, position, new_aa = parse_mutation_label(picked)
                with st.spinner("Designing…"):
                    result = design_mutation_primers(
                        sequence, position, orig_aa, new_aa,
                        orf_start=int(orf_start),
                        tm_range=(tm_min, tm_max),
                        window_bp=(int(win_near), int(win_far)),
                    )
                _render_result(result, key_prefix="find_")
        if failing:
            with st.expander(f"{len(failing)} position(s) not designable"):
                st.write(", ".join(f"{from_aa}{p}{to_aa}" for p, _ in failing))
