# src/nufft_biot/forward.py
from __future__ import annotations
import jax.numpy as jnp

from .types import BoxParams
from .geometry import geom_from_x
from .current import compute_J_components_from_x_and_geom
from .field import B_from_nodes_and_J


def geom_and_J_from_x(
    x: jnp.ndarray,
    *,
    I: float,
    major_radius: float,
    minor_radius: float,
    N_rho: int,
    N_theta: int,
    N_zeta: int,
):
    X, Y, Z, rho_nodes, theta_nodes, zeta, w = geom_from_x(
        x,
        N_rho=N_rho,
        N_theta=N_theta,
        N_zeta=N_zeta,
        major_radius=major_radius,
        minor_radius=minor_radius,
    )

    R = jnp.sqrt(X**2 + Y**2)

    Jx, Jy, Jz = compute_J_components_from_x_and_geom(
        x,
        R,
        Z,
        rho_nodes,
        theta_nodes,
        zeta,
        I=I,
        major_radius=major_radius,
        minor_radius=minor_radius,
    )

    return X, Y, Z, Jx, Jy, Jz, w


def forward_B(
    x: jnp.ndarray,
    box: BoxParams,
    *,
    I: float,
    major_radius: float,
    minor_radius: float,
    N_rho: int = 8,
    N_theta: int = 16,
    N_zeta: int = 32,
):
    X, Y, Z, Jx, Jy, Jz, w = geom_and_J_from_x(
        x,
        I=I,
        major_radius=major_radius,
        minor_radius=minor_radius,
        N_rho=N_rho,
        N_theta=N_theta,
        N_zeta=N_zeta,
    )

    return B_from_nodes_and_J(
        X,
        Y,
        Z,
        Jx,
        Jy,
        Jz,
        w,
        box,
    )

