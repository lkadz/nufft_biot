from __future__ import annotations
import jax.numpy as jnp

from .types import BoxParams


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
