"""
Streamlit web app for Gibson SDM primer design.

Run locally:
    streamlit run app.py

This is a thin UI layer only — all the actual design logic lives in
pipeline.py / primer_bc.py / primer_ad.py / assembly.py / restriction_utils.py
/ codon_utils.py, unchanged. The app just calls those same functions.
"""

import io

import streamlit as st
from Bio import SeqIO
from Bio.Seq import Seq

from pipeline import design_mutation_primers, find_all_positions, parse_mutation_label
from primer_ad import NEB_CATALOG

st.set_page_config(page_title="Gibson SDM Primer Designer", layout="wide")


# ---------------------------------------------------------------------------
# Helpers (mirrors sdm_design.py's CLI logic, adapted for Streamlit widgets)
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
        # fall back to any ATG at all
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
    if info.get("hf"):
        tag = f"NEB {info['hf']} (HF)"
    else:
        tag = f"NEB {info['cat']}"
    if info.get("note"):
        tag += f"  ⚠ {info['note']}"
    return tag


# ---------------------------------------------------------------------------
# Sidebar: sequence input (shared across both modes)
# ---------------------------------------------------------------------------

st.title("🧬 Gibson SDM Primer Designer")
st.caption("Design Gibson site-directed mutagenesis primers from a FASTA sequence.")

with st.sidebar:
    st.header("1. Sequence")
    upload = st.file_uploader("Upload a FASTA file", type=["fasta", "fa", "txt"])
    pasted = st.text_area("…or paste FASTA text", height=150,
                           placeholder=">my_construct\nATGGCT...")

    sequence = None
    if upload is not None:
        sequence = _read_fasta_text(upload.getvalue().decode("utf-8", errors="replace"))
    elif pasted.strip():
        sequence = _read_fasta_text(pasted)

    if sequence:
        st.success(f"Loaded {len(sequence)} bp")

    st.header("2. ORF start")
    orf_start = None
    if sequence:
        candidates = _find_orf_candidates(sequence)
        if candidates:
            labels = [f"nt {start}  ({length} aa){'  ◀ longest' if i == 0 else ''}"
                      for i, (start, length) in enumerate(candidates)]
            labels.append("Custom position…")
            choice = st.selectbox("Pick the ORF start codon (ATG)", labels, index=0)
            if choice == "Custom position…":
                orf_start = st.number_input("Custom ORF start (0-based nt)",
                                             min_value=0, max_value=len(sequence) - 1, value=0)
            else:
                orf_start = candidates[labels.index(choice)][0]
        else:
            st.error("No ATG found in this sequence.")

    st.header("3. Design settings")
    tm_min = st.number_input("Tm min (°C)", value=48.0, step=1.0)
    tm_max = st.number_input("Tm max (°C)", value=54.0, step=1.0)
    win_near = st.number_input("A/D window near (bp, per side)", value=350, step=50)
    win_far = st.number_input("A/D window far (bp, per side)", value=500, step=50)


# ---------------------------------------------------------------------------
# Main panel: two modes
# ---------------------------------------------------------------------------

if not sequence or orf_start is None:
    st.info("Upload or paste a FASTA sequence in the sidebar to get started.")
    st.stop()

mode = st.radio("Mode", ["Design a specific mutation", "Find all positions for an AA change"],
                 horizontal=True)

# ---------------------------------------------------------------------------
def _render_result(result, key_prefix=""):
    if result.errors:
        for e in result.errors:
            st.error(e)
        return

    for w in result.warnings:
        st.warning(w)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mutation", result.mutation_label)
        st.write(f"Codon: `{result.original_codon}` → `{result.new_codon}`")
        st.write(f"ORF start: nt {result.orf_start_detected} (0-based)")
        st.write(f"Mutation site: nt {result.mutation_nt_position} (0-based)")
    with col2:
        if result.diagnostic:
            d = result.diagnostic
            src = "(from mutation)" if d.source == "mutation" else \
                  (f"(silent mutation: aa{d.silent_aa_index} "
                   f"{d.silent_original_codon}→{d.silent_new_codon})")
            st.write(f"**Diagnostic site:** {d.enzyme} — {d.effect.upper()} {src}")
        else:
            st.write("**Diagnostic site:** none found")

    st.subheader("Primers (5'→3')")
    rows = []
    if result.primer_A:
        pa = result.primer_A
        rows.append(["A", pa.sequence, f"{pa.tm:.0f}°C",
                     f"{pa.enzyme} {_neb_tag(pa.enzyme)}  cut {pa.cut_pos}"])
    if result.primer_B:
        pb = result.primer_B
        rows.append(["B", pb.sequence, f"{pb.tm:.0f}°C", "sense / forward"])
    if result.primer_C:
        pc = result.primer_C
        rows.append(["C", pc.sequence,
                     f"{pc.tm:.0f}°C full / {pc.tm_anneal:.0f}°C anneal",
                     "antisense / reverse"])
    if result.primer_D:
        pd = result.primer_D
        rows.append(["D", pd.sequence, f"{pd.tm:.0f}°C",
                     f"{pd.enzyme} {_neb_tag(pd.enzyme)}  cut {pd.cut_pos}"])
    st.table(
        {"Primer": [r[0] for r in rows],
         "Sequence": [r[1] for r in rows],
         "Tm": [r[2] for r in rows],
         "Notes": [r[3] for r in rows]}
    )

    st.write(f"**Overlap:** `{result.overlap_seq}` "
             f"({len(result.overlap_seq)} bp, Tm={result.overlap_tm:.0f}°C)")

    if result.frag_ad_bp:
        st.write(f"**Fragments:** ab={result.frag_ab_bp} bp, "
                 f"cd={result.frag_cd_bp} bp, assembled={result.frag_ad_bp} bp")

    st.subheader("Verification")
    c1, c2, c3 = st.columns(3)
    c1.metric("Translation", "PASS ✓" if result.translation_passed else "FAIL ✗")
    c2.metric("Restriction site",
              "PASS ✓" if result.restriction_passed else
              ("SKIP" if result.restriction_passed is None else "FAIL ✗"))
    c3.metric("Overall", "PASS ✓" if result.overall_passed else "FAIL ✗")

    report_text = (
        f"Mutation: {result.mutation_label}\n"
        f"Codon: {result.original_codon} -> {result.new_codon}\n"
        f"ORF start: {result.orf_start_detected}\n\n"
        + "\n".join(f"{r[0]}: {r[1]}  Tm={r[2]}  {r[3]}" for r in rows)
        + f"\n\nOverlap: {result.overlap_seq}\n"
        f"Overall: {'PASS' if result.overall_passed else 'FAIL'}\n"
    )
    st.download_button("Download report (.txt)", report_text,
                        file_name=f"{result.mutation_label}_report.txt",
                        key=f"{key_prefix}dl")


if mode == "Design a specific mutation":
    st.subheader("Design a specific mutation")
    label = st.text_input("Mutation (e.g. K255E)", placeholder="K255E")
    if st.button("Design primers", type="primary") and label:
        try:
            orig_aa, position, new_aa = parse_mutation_label(label)
        except ValueError as exc:
            st.error(str(exc))
        else:
            with st.spinner("Designing primers…"):
                result = design_mutation_primers(
                    sequence=sequence,
                    target_position=position,
                    original_aa=orig_aa,
                    new_aa=new_aa,
                    orf_start=int(orf_start),
                    tm_range=(tm_min, tm_max),
                    window_bp=(int(win_near), int(win_far)),
                )
            _render_result(result)

else:
    st.subheader("Find all designable positions for an amino-acid change")
    col1, col2 = st.columns(2)
    from_aa = col1.text_input("From (single-letter AA)", value="K", max_chars=1).upper()
    to_aa = col2.text_input("To (single-letter AA)", value="E", max_chars=1).upper()

    if st.button("Find positions", type="primary") and from_aa and to_aa:
        hits = find_all_positions(sequence, int(orf_start), from_aa)
        if not hits:
            st.warning(f"No {from_aa} residues found in this ORF.")
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

            st.success(f"{len(working)} of {len(hits)} position(s) designable")
            if working:
                st.table({
                    "Position": [f"{from_aa}{p}{to_aa}" for p, _, _ in working],
                    "Codon": [c for _, c, _ in working],
                    "Note": [n for _, _, n in working],
                })
                picked = st.selectbox(
                    "Pick a position to see the full design",
                    [f"{from_aa}{p}{to_aa}" for p, _, _ in working],
                )
                if picked and st.button("Show full design for selected position"):
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
                with st.expander(f"{len(failing)} position(s) filtered out"):
                    st.write(", ".join(f"{from_aa}{p}{to_aa}" for p, _ in failing))
