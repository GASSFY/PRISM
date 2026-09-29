"""Factorial residuals for the 8-cell Vision / Projector / LLM design.

A cell name is a subset of {V, P, L}, written in that letter order.
The empty name is the full-precision cell.

Losses are defined so that larger means worse, and the FP cell is the zero
of the loss (measured error against FP). Pairwise interaction then collapses
to L(AB) - L(A) - L(B). The three-way term keeps the full inclusion-exclusion
so that a nonzero value is not just the sum of the three pairs.
"""
from __future__ import annotations

COMPONENTS = ("V", "P", "L")
CELL_NAMES = ("", "V", "P", "L", "VP", "VL", "PL", "VPL")


def canon(letters: str) -> str:
    return "".join(ch for ch in COMPONENTS if ch in letters)


def pairwise(cells: dict[str, float], a: str, b: str) -> float:
    """I(A,B) = L(A∪B) - L(A) - L(B) + L(∅)."""
    both = canon(a + b)
    return cells[both] - cells[canon(a)] - cells[canon(b)] + cells[""]


def three_way(cells: dict[str, float]) -> float:
    """I(V,P,L) after removing the three pairwise contributions."""
    return (
        cells["VPL"]
        - cells["VP"]
        - cells["VL"]
        - cells["PL"]
        + cells["V"]
        + cells["P"]
        + cells["L"]
        - cells[""]
    )


def interaction_row(cells: dict[str, float]) -> dict[str, float]:
    missing = [name for name in CELL_NAMES if name not in cells]
    if missing:
        raise KeyError(f"missing cells: {missing}")
    return {
        "VP": pairwise(cells, "V", "P"),
        "VL": pairwise(cells, "V", "L"),
        "PL": pairwise(cells, "P", "L"),
        "VPL": three_way(cells),
    }


def _self_check() -> None:
    additive = {name: float(len(name)) for name in CELL_NAMES}
    # len("")=0, V=1, P=1, L=1, VP=2, ... so every interaction is 0.
    row = interaction_row(additive)
    assert all(abs(v) < 1e-12 for v in row.values()), row

    super_add = dict(additive)
    super_add["VL"] = 10.0
    row = interaction_row(super_add)
    assert abs(row["VL"] - (10.0 - 1.0 - 1.0)) < 1e-12, row
    assert abs(row["VP"]) < 1e-12 and abs(row["PL"]) < 1e-12

    only_triple = dict(additive)
    only_triple["VPL"] = 9.0
    row = interaction_row(only_triple)
    # pairs unchanged and zero; three-way picks up the extra 9 - 2.
    assert abs(row["VPL"] - (9.0 - 2.0 - 2.0 - 2.0 + 1.0 + 1.0 + 1.0)) < 1e-12, row
    print("interaction self-check ok", row)


if __name__ == "__main__":
    _self_check()
