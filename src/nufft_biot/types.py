from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class BoxParams:
    Lx: float
    Ly: float
    Lz: float
    Nx: int
    Ny: int
    Nz: int
