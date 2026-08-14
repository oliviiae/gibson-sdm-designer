"""
Part 1 — Codon mutation utilities.

find_codon_and_mutate(dna_seq, orf_start, aa_position, new_aa)
  -> (mutated_seq, changed_positions, original_codon, new_codon)

aa_position is 1-based (matches standard amino-acid numbering).
orf_start   is 0-based index into dna_seq where the ORF begins.
"""

from Bio.Data import CodonTable
from Bio.Seq import Seq


# Standard genetic code (table 1)
_CODON_TABLE = CodonTable.standard_dna_table


def _codons_for_aa(aa: str) -> list[str]:
    """Return all DNA codons that encode the given single-letter amino acid."""
    if aa.upper() == '*':
        return list(_CODON_TABLE.stop_codons)
    return [codon for codon, residue in _CODON_TABLE.forward_table.items()
            if residue == aa.upper()]


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def _changed_positions(original: str, mutated: str) -> list[int]:
    """0-based positions within the codon that differ."""
    return [i for i, (a, b) in enumerate(zip(original, mutated)) if a != b]


def ranked_codon_options(original_codon: str, new_aa: str) -> list[str]:
    """
    All codons encoding new_aa that differ from original_codon, ranked by
    fewest nucleotide changes first (ties broken alphabetically for
    determinism). The first element is what find_codon_and_mutate uses.

    Exposed publicly so callers can try alternate codons for the SAME target
    amino acid — per protocol step 1a: "change the nucleotide again" is the
    first fallback when the minimal-change codon doesn't yield a usable
    diagnostic site, tried before resorting to an adjacent silent mutation.
    """
    candidates = _codons_for_aa(new_aa)
    candidates = [c for c in candidates if c != original_codon.upper()]
    return sorted(candidates, key=lambda c: (_hamming(original_codon, c), c))


def apply_codon(
    dna_seq: str,
    orf_start: int,
    aa_position: int,
    specific_codon: str,
) -> tuple[str, list[int], str, str]:
    """
    Like find_codon_and_mutate, but swaps in a caller-chosen codon instead of
    auto-picking the fewest-change one. Used to try alternate synonymous
    codon options for the same target amino acid.
    """
    dna_seq = dna_seq.upper()
    codon_start = orf_start + (aa_position - 1) * 3
    original_codon = dna_seq[codon_start: codon_start + 3]
    specific_codon = specific_codon.upper()

    changes = _changed_positions(original_codon, specific_codon)
    abs_positions = [codon_start + i for i in changes]
    mutated_seq = dna_seq[:codon_start] + specific_codon + dna_seq[codon_start + 3:]
    return mutated_seq, abs_positions, original_codon, specific_codon


def find_codon_and_mutate(
    dna_seq: str,
    orf_start: int,
    aa_position: int,
    new_aa: str,
) -> tuple[str, list[int], str, str]:
    """
    Locate the codon for aa_position (1-based) within the ORF and replace it
    with a codon for new_aa, preferring the fewest nucleotide changes.

    Returns
    -------
    mutated_seq      : full DNA string with the codon swapped
    changed_positions: 0-based absolute positions in dna_seq that were altered
    original_codon   : the codon before mutation
    new_codon        : the codon after mutation
    """
    dna_seq = dna_seq.upper()
    # Codon start (0-based) in the full sequence
    codon_start = orf_start + (aa_position - 1) * 3
    original_codon = dna_seq[codon_start: codon_start + 3]

    if len(original_codon) != 3:
        raise ValueError(
            f"Codon at aa position {aa_position} extends beyond the supplied sequence."
        )

    # Translate original codon to verify identity
    original_aa = str(Seq(original_codon).translate())
    if original_aa == new_aa.upper():
        raise ValueError(
            f"Original codon {original_codon} already encodes {new_aa}."
        )

    # Find best replacement codon (fewest changes, then alphabetical for determinism)
    candidates = _codons_for_aa(new_aa)
    if not candidates:
        raise ValueError(f"No codons found for amino acid '{new_aa}'.")

    best_codon = min(candidates, key=lambda c: (_hamming(original_codon, c), c))
    changes = _changed_positions(original_codon, best_codon)
    abs_positions = [codon_start + i for i in changes]

    mutated_seq = dna_seq[:codon_start] + best_codon + dna_seq[codon_start + 3:]

    return mutated_seq, abs_positions, original_codon, best_codon
