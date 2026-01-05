from __future__ import annotations
import jax.numpy as jnp

from .geometry import geom_from_x


def torus_volume_current(
    *,
    I: float,
    major_radius: float,
    minor_radius: float,
    N_rho: int,
    N_theta: int,
    N_zeta: int,
):
    X, Y, Z, rho_nodes, theta_nodes, zeta, w = geom_from_x(
        jnp.zeros((1,), dtype=jnp.float64),
        N_rho=N_rho,
        N_theta=N_theta,
        N_zeta=N_zeta,
        major_radius=major_radius,
        minor_radius=minor_radius,
    )

    R = jnp.sqrt(X**2 + Y**2)

    inside = (R - major_radius) ** 2 + Z**2 <= minor_radius**2
    J_phi = jnp.where(inside, I / (jnp.pi * minor_radius**2), 0.0)

    Jx = -J_phi * jnp.sin(zeta)
    Jy = J_phi * jnp.cos(zeta)
    Jz = jnp.zeros_like(Jx)

    return X, Y, Z, Jx, Jy, Jz, w


def torus_axis_filament_current(
    *,
    I: float,
    major_radius: float,
    N_zeta: int,
):
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi, N_zeta, endpoint=False)

    X = major_radius * jnp.cos(zeta)
    Y = major_radius * jnp.sin(zeta)
    Z = jnp.zeros_like(zeta)

    Jx = -I * jnp.sin(zeta)
    Jy = I * jnp.cos(zeta)
    Jz = jnp.zeros_like(zeta)

    w = (2.0 * jnp.pi * major_radius / N_zeta) * jnp.ones_like(zeta)

    return X, Y, Z, Jx, Jy, Jz, w
