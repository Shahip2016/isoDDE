"""Chemical component dictionary (CCD) handling and reference values.

Provides ideal bond lengths, angles, and element properties used for
ligand violation detection (Section 4.3) and featurization.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Element properties
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ElementProperties:
    """Physical properties of a chemical element."""
    symbol: str
    atomic_number: int
    mass: float
    van_der_waals_radius: float  # Å
    covalent_radius: float  # Å


# Van der Waals and covalent radii from standard references
ELEMENT_PROPERTIES: dict[str, ElementProperties] = {
    "H":  ElementProperties("H",  1,   1.008,  1.20, 0.31),
    "B":  ElementProperties("B",  5,  10.81,   1.92, 0.84),
    "C":  ElementProperties("C",  6,  12.011,  1.70, 0.76),
    "N":  ElementProperties("N",  7,  14.007,  1.55, 0.71),
    "O":  ElementProperties("O",  8,  15.999,  1.52, 0.66),
    "F":  ElementProperties("F",  9,  18.998,  1.47, 0.57),
    "Si": ElementProperties("Si", 14, 28.086,  2.10, 1.11),
    "P":  ElementProperties("P",  15, 30.974,  1.80, 1.07),
    "S":  ElementProperties("S",  16, 32.06,   1.80, 1.05),
    "Cl": ElementProperties("Cl", 17, 35.45,   1.75, 1.02),
    "Se": ElementProperties("Se", 34, 78.96,   1.90, 1.20),
    "Br": ElementProperties("Br", 35, 79.904,  1.85, 1.20),
    "I":  ElementProperties("I",  53, 126.90,  1.98, 1.39),
}


# ---------------------------------------------------------------------------
# Ideal bond lengths (Å) — from CCD reference
# ---------------------------------------------------------------------------

IDEAL_BOND_LENGTHS: dict[tuple[str, str, str], float] = {
    # (element1, element2, bond_type) -> length
    ("C", "C", "SINGLE"):   1.54,
    ("C", "C", "DOUBLE"):   1.34,
    ("C", "C", "TRIPLE"):   1.20,
    ("C", "C", "AROMATIC"): 1.40,
    ("C", "N", "SINGLE"):   1.47,
    ("C", "N", "DOUBLE"):   1.29,
    ("C", "N", "AROMATIC"): 1.34,
    ("C", "O", "SINGLE"):   1.43,
    ("C", "O", "DOUBLE"):   1.23,
    ("C", "S", "SINGLE"):   1.82,
    ("C", "S", "DOUBLE"):   1.60,
    ("C", "F", "SINGLE"):   1.35,
    ("C", "Cl", "SINGLE"):  1.77,
    ("C", "Br", "SINGLE"):  1.94,
    ("C", "I", "SINGLE"):   2.14,
    ("N", "N", "SINGLE"):   1.45,
    ("N", "N", "DOUBLE"):   1.25,
    ("N", "O", "SINGLE"):   1.40,
    ("N", "O", "DOUBLE"):   1.21,
    ("O", "P", "SINGLE"):   1.63,
    ("O", "P", "DOUBLE"):   1.48,
    ("O", "S", "DOUBLE"):   1.43,
    ("S", "S", "SINGLE"):   2.05,
    ("C", "P", "SINGLE"):   1.84,
    ("C", "H", "SINGLE"):   1.09,
    ("N", "H", "SINGLE"):   1.01,
    ("O", "H", "SINGLE"):   0.96,
    ("S", "H", "SINGLE"):   1.34,
}

# ---------------------------------------------------------------------------
# Ideal bond angles (degrees)
# ---------------------------------------------------------------------------

IDEAL_BOND_ANGLES: dict[str, float] = {
    # Hybridisation -> ideal angle
    "sp3": 109.5,
    "sp2": 120.0,
    "sp":  180.0,
}


def get_ideal_bond_length(
    elem1: str, elem2: str, bond_type: str
) -> float | None:
    """Look up ideal bond length from CCD reference.

    Parameters
    ----------
    elem1, elem2 : str
        Element symbols.
    bond_type : str
        One of SINGLE, DOUBLE, TRIPLE, AROMATIC.

    Returns
    -------
    float or None
        Ideal bond length in Å, or None if not found.
    """
    key = (elem1, elem2, bond_type)
    if key in IDEAL_BOND_LENGTHS:
        return IDEAL_BOND_LENGTHS[key]
    # Try reversed
    key_rev = (elem2, elem1, bond_type)
    return IDEAL_BOND_LENGTHS.get(key_rev)


def get_vdw_radius(element: str) -> float:
    """Get van der Waals radius for an element."""
    props = ELEMENT_PROPERTIES.get(element)
    if props is None:
        return 1.70  # Default (carbon-like)
    return props.van_der_waals_radius


def get_covalent_radius(element: str) -> float:
    """Get covalent radius for an element."""
    props = ELEMENT_PROPERTIES.get(element)
    if props is None:
        return 0.76  # Default
    return props.covalent_radius
