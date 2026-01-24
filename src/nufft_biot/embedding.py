from __future__ import annotations
import jax.numpy as jnp

from .types import BoxParams


def make_optimal_box(
    X: jnp.ndarray, 
    Y: jnp.ndarray, 
    Z: jnp.ndarray, 
    n_cells: int = 256, 
    padding: float = 2.0
) -> BoxParams:
  
    xmin, xmax = float(X.min()), float(X.max())
    ymin, ymax = float(Y.min()), float(Y.max())
    zmin, zmax = float(Z.min()), float(Z.max())
    
    Lx_geom = xmax - xmin
    Ly_geom = ymax - ymin
    Lz_geom = zmax - zmin
    
    max_L = max(Lx_geom, Ly_geom, Lz_geom)
    
    L_final = max_L * padding
    
    print(f"Geometry Extent: [{Lx_geom:.2f}, {Ly_geom:.2f}, {Lz_geom:.2f}]")
    print(f"Auto-sized Box:  {L_final:.2f} (cubic)")
    
    return BoxParams(
        Lx=L_final,
        Ly=L_final,
        Lz=L_final,
        Nx=n_cells,
        Ny=n_cells,
        Nz=n_cells,
    )


def embed_geometry_in_box(X, Y, Z, box: BoxParams):
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()

    Lxg = xmax - xmin
    Lyg = ymax - ymin
    Lzg = zmax - zmin

    shift_x = 0.5 * (box.Lx - Lxg) - xmin
    shift_y = 0.5 * (box.Ly - Lyg) - ymin
    shift_z = 0.5 * (box.Lz - Lzg) - zmin

    Xb = X + shift_x
    Yb = Y + shift_y
    Zb = Z + shift_z

    center = jnp.array(
        [
            0.5 * (xmin + xmax) + shift_x,
            0.5 * (ymin + ymax) + shift_y,
            0.5 * (zmin + zmax) + shift_z,
        ],
        dtype=X.dtype,
    )

    return Xb, Yb, Zb, center
