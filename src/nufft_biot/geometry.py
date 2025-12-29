#src/nufft_biot/geometry.py
from __future__ import annotations
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def make_torus_nodes(
    *,
    N_rho: int,
    N_theta: int,
    N_zeta: int,
):
    rho_nodes = (jnp.arange(N_rho) + 0.5) / N_rho
    theta_nodes = (jnp.arange(N_theta) + 0.5) * (2.0 * jnp.pi / N_theta)
    zeta_nodes = (jnp.arange(N_zeta)) * (2.0 * jnp.pi / N_zeta)

    rho, theta, zeta = jnp.meshgrid(
        rho_nodes, theta_nodes, zeta_nodes, indexing="ij"
    )

    return rho, theta, zeta


def torus_geometry(
    rho: jnp.ndarray,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
    *,
    major_radius: float,
    minor_radius: float,
):
    R = major_radius + minor_radius * rho * jnp.cos(theta)
    Z = minor_radius * rho * jnp.sin(theta)

    X = R * jnp.cos(zeta)
    Y = R * jnp.sin(zeta)

    return X, Y, Z, R, Z


def torus_volume_weights(
    rho: jnp.ndarray,
    theta: jnp.ndarray,
    *,
    major_radius: float,
    minor_radius: float,
    N_rho: int,
    N_theta: int,
    N_zeta: int,
):
    drho = 1.0 / N_rho
    dtheta = 2.0 * jnp.pi / N_theta
    dzeta = 2.0 * jnp.pi / N_zeta

    w = (
        minor_radius**2
        * rho
        * (major_radius + minor_radius * rho * jnp.cos(theta))
        * drho
        * dtheta
        * dzeta
    )

    return w


def geom_from_x(
    x: jnp.ndarray,
    *,
    N_rho: int = 8,
    N_theta: int = 16,
    N_zeta: int = 32,
    minor_radius: float,
    major_radius: float,
):
    rho, theta, zeta = make_torus_nodes(
        N_rho=N_rho,
        N_theta=N_theta,
        N_zeta=N_zeta,
    )

    X, Y, Z, R, Zc = torus_geometry(
        rho,
        theta,
        zeta,
        major_radius=major_radius,
        minor_radius=minor_radius,
    )

    w = torus_volume_weights(
        rho,
        theta,
        major_radius=major_radius,
        minor_radius=minor_radius,
        N_rho=N_rho,
        N_theta=N_theta,
        N_zeta=N_zeta,
    )

    return (
        X.reshape(-1),
        Y.reshape(-1),
        Z.reshape(-1),
        rho.reshape(-1),
        theta.reshape(-1),
        zeta.reshape(-1),
        w.reshape(-1),
    )


def compute_loop_radius(x: jnp.ndarray) -> float:
    return x[0]
