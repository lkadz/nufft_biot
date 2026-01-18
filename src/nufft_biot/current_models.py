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
    use_nonuniform_grid: bool = False,
):
    X, Y, Z, rho_nodes, theta_nodes, zeta, w = geom_from_x(
        jnp.zeros((1,), dtype=jnp.float64),
        N_rho=N_rho,
        N_theta=N_theta,
        N_zeta=N_zeta,
        major_radius=major_radius,
        minor_radius=minor_radius,
        use_nonuniform_grid=use_nonuniform_grid,
    )

    R = jnp.sqrt(X**2 + Y**2)

    rho = jnp.sqrt((R - major_radius)**2 + Z**2) / minor_radius
    inside = rho <= 1.0

    alpha = 10.0 
    f = jnp.where(inside, (1.0 - rho**2)**alpha, 0.0)

    J0 = I / (jnp.pi * minor_radius**2 * (1.0 / (alpha + 1.0)))

    J_phi = J0 * f

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


def tokamak_like_current(
    *,
    I: float,
    major_radius: float,
    minor_radius: float,
    N_rho: int,
    N_theta: int,
    N_zeta: int,
    alpha: float = 1.0,
    A0: float = 0.0,
    p: int = 2,
    m: int = 1,
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
    dR = R - major_radius
    s2 = dR**2 + Z**2
    s = jnp.sqrt(s2)

    rho = s / minor_radius
    inside = rho <= 1.0

    theta = jnp.arctan2(Z, dR)
    phi = zeta

    f = jnp.where(inside, (1.0 - rho**2) ** alpha, 0.0)

    drho = 1.0 / N_rho
    dtheta = 2.0 * jnp.pi / N_theta
    w_cs = minor_radius**2 * rho_nodes * drho * dtheta

    I_est = jnp.sum(f * w_cs) / N_zeta
    J0 = jnp.where(I_est > 0.0, I / I_est, 0.0)

    Jphi = J0 * f

    if A0 == 0.0:
        JR = jnp.zeros_like(Jphi)
        JZ = jnp.zeros_like(Jphi)
    else:
        rho_safe = jnp.where(rho > 1e-14, rho, 1e-14)
        s2_safe = jnp.where(s2 > 1e-28, s2, 1e-28)
        R_safe = jnp.where(R > 1e-14, R, 1e-14)

        rho_R = dR / (minor_radius**2 * rho_safe)
        rho_Z = Z / (minor_radius**2 * rho_safe)

        theta_R = -Z / s2_safe
        theta_Z = dR / s2_safe

        g = jnp.where(inside, rho**p * (1.0 - rho**2) ** 2, 0.0)
        dg_drho = jnp.where(
            inside,
            p * rho_safe ** (p - 1) * (1.0 - rho**2) ** 2
            + rho**p * 2.0 * (1.0 - rho**2) * (-2.0 * rho),
            0.0,
        )

        Aphi = A0 * g * jnp.cos(m * theta)
        dA_dtheta = A0 * g * (-m) * jnp.sin(m * theta)
        dA_drho = A0 * dg_drho * jnp.cos(m * theta)

        dA_dR = dA_drho * rho_R + dA_dtheta * theta_R
        dA_dZ = dA_drho * rho_Z + dA_dtheta * theta_Z

        RA = R * Aphi
        dRA_dR = Aphi + R * dA_dR
        dRA_dZ = R * dA_dZ

        JR = (1.0 / R_safe) * dRA_dZ
        JZ = -(1.0 / R_safe) * dRA_dR

    Jx = JR * jnp.cos(phi) - Jphi * jnp.sin(phi)
    Jy = JR * jnp.sin(phi) + Jphi * jnp.cos(phi)
    Jz = JZ

    return X, Y, Z, Jx, Jy, Jz, w
