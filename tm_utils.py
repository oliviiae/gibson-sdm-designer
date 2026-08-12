"""
Part 2 — Tm calculation and primer boundary walking.

simple_tm(seq)
  -> float  (sum rule: G/C = 4°C, A/T = 2°C)

walk_to_tm(seq, anchor_pos, direction, tm_range, must_end_gc)
  -> (primer_seq, start, end, tm)

anchor_pos is the 0-based position to start walking from (inclusive).
direction  is +1 (walk right / toward 3') or -1 (walk left / toward 5').
The walk *always includes* anchor_pos and expands outward until the
target Tm window is first entered, then continues while still in window,
stopping when adding the next base would leave the window OR the sequence
ends.  If must_end_gc=True, the walk may extend one extra base to land on G/C.
"""


def simple_tm(seq: str) -> float:
    """Wallace / nearest-neighbour-free Tm estimate."""
    seq = seq.upper()
    return sum(4 if b in "GC" else 2 for b in seq if b in "ACGT")


def walk_to_tm(
    seq: str,
    anchor_pos: int,
    direction: int,
    tm_range: tuple[float, float] = (48.0, 54.0),
    must_end_gc: bool = True,
    max_len: int = 60,
) -> tuple[str, int, int, float]:
    """
    Extend a primer starting from anchor_pos in the given direction until the
    cumulative Tm falls within tm_range and (optionally) the terminal base is G or C.

    Parameters
    ----------
    seq        : full DNA sequence (sense strand, 5'→3')
    anchor_pos : 0-based index; the primer always contains this base
    direction  : +1 extend toward higher indices, -1 toward lower indices
    tm_range   : (min_tm, max_tm) inclusive
    must_end_gc: if True, extend past the window if the current end is A/T
    max_len    : safety cap on primer length

    Returns
    -------
    primer_seq : the primer sequence (always written 5'→3')
    start      : 0-based start index in seq (inclusive)
    end        : 0-based end index in seq (exclusive, Python-slice style)
    tm         : Tm of the returned primer
    """
    seq = seq.upper()
    tm_lo, tm_hi = tm_range

    left = anchor_pos
    right = anchor_pos + 1  # slice notation: seq[left:right]

    def current_primer():
        return seq[left:right]

    def end_base():
        """The terminal base in the extension direction."""
        return seq[right - 1] if direction == 1 else seq[left]

    for _ in range(max_len):
        tm = simple_tm(current_primer())
        in_window = tm_lo <= tm <= tm_hi
        gc_ok = end_base() in "GC"

        if in_window and (not must_end_gc or gc_ok):
            break

        # Decide which side to extend
        if direction == 1:
            if right >= len(seq):
                break
            right += 1
        else:
            if left <= 0:
                break
            left -= 1
    else:
        # hit max_len — return what we have
        pass

    primer = seq[left:right]
    return primer, left, right, simple_tm(primer)
