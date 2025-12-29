#src/nufft_biot/current.py
from __future__ import annotations
import jax.numpy as jnp


def compute_J_components_from_x_and_geom(
    x: jnp.ndarray,
    R: jnp.ndarray,
    Z: jnp.ndarray,
    rho_nodes: jnp.ndarray,
    theta_nodes: jnp.ndarray,
    zeta: jnp.ndarray,
    *,
    I: float,
    minor_radius: float,
    major_radius: float,
):
    inside = (R - major_radius) ** 2 + Z ** 2 <= minor_radius ** 2

    J_phi = jnp.where(
        inside,
        I / (jnp.pi * minor_radius**2),
        0.0,
    )

    Jx = -J_phi * jnp.sin(zeta)
    Jy =  J_phi * jnp.cos(zeta)
    Jz = jnp.zeros_like(Jx)

    return Jx, Jy, Jz



