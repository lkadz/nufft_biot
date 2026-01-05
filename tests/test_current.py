import jax.numpy as jnp

from nufft_biot.current_models import (
    torus_volume_current,
    torus_axis_filament_current,
)


def test_volume_current_support_inside_outside():
    R0 = 1.0
    a = 0.3
    I = 2.5

    X, Y, Z, Jx, Jy, Jz, w = torus_volume_current(
        I=I,
        major_radius=R0,
        minor_radius=a,
        N_rho=8,
        N_theta=16,
        N_zeta=32,
    )

    R = jnp.sqrt(X**2 + Y**2)

    inside = (R - R0) ** 2 + Z**2 <= a**2

    assert jnp.all(Jz == 0.0)
    assert jnp.all((Jx[~inside] == 0.0) & (Jy[~inside] == 0.0))
    assert jnp.any(Jx[inside]**2 + Jy[inside]**2 > 0.0)


def test_volume_current_purely_toroidal():
    R0 = 1.2
    a = 0.4
    I = 1.0

    X, Y, Z, Jx, Jy, _, _ = torus_volume_current(
        I=I,
        major_radius=R0,
        minor_radius=a,
        N_rho=6,
        N_theta=12,
        N_zeta=64,
    )

    zeta = jnp.arctan2(Y, X)

    radial_dot = Jx * jnp.cos(zeta) + Jy * jnp.sin(zeta)

    assert jnp.allclose(radial_dot, 0.0, atol=1e-12)


def test_volume_current_uniform_magnitude():
    R0 = 0.9
    a = 0.25
    I = 3.0

    X, Y, Z, Jx, Jy, _, _ = torus_volume_current(
        I=I,
        major_radius=R0,
        minor_radius=a,
        N_rho=10,
        N_theta=20,
        N_zeta=40,
    )

    R = jnp.sqrt(X**2 + Y**2)
    inside = (R - R0) ** 2 + Z**2 <= a**2

    Jmag = jnp.sqrt(Jx**2 + Jy**2)
    J_expected = I / (jnp.pi * a**2)

    assert jnp.allclose(Jmag[inside], J_expected, rtol=1e-12)
    assert jnp.all(Jmag[~inside] == 0.0)


def test_volume_current_total_current():
    R0 = 1.1
    a = 0.35
    I = 2.0

    X, Y, Z, Jx, Jy, _, w = torus_volume_current(
        I=I,
        major_radius=R0,
        minor_radius=a,
        N_rho=24,
        N_theta=48,
        N_zeta=1,
    )

    zeta = jnp.arctan2(Y, X)
    Jphi = -Jx * jnp.sin(zeta) + Jy * jnp.cos(zeta)

    I_num = jnp.sum(Jphi * w) / (2.0 * jnp.pi * R0)

    assert jnp.allclose(I_num, I, rtol=5e-3)


def test_filament_geometry():
    R0 = 1.0
    I = 2.0

    X, Y, Z, Jx, Jy, Jz, w = torus_axis_filament_current(
        I=I,
        major_radius=R0,
        N_zeta=128,
    )

    R = jnp.sqrt(X**2 + Y**2)

    assert jnp.allclose(R, R0)
    assert jnp.all(Z == 0.0)
    assert jnp.all(Jz == 0.0)


def test_filament_purely_toroidal():
    R0 = 1.3
    I = 1.5

    X, Y, _, Jx, Jy, _, _ = torus_axis_filament_current(
        I=I,
        major_radius=R0,
        N_zeta=256,
    )

    zeta = jnp.arctan2(Y, X)

    radial_dot = Jx * jnp.cos(zeta) + Jy * jnp.sin(zeta)

    assert jnp.allclose(radial_dot, 0.0, atol=1e-12)


def test_filament_total_current():
    R0 = 0.8
    I = 4.0

    X, Y, _, Jx, Jy, _, w = torus_axis_filament_current(
        I=I,
        major_radius=R0,
        N_zeta=512,
    )

    zeta = jnp.arctan2(Y, X)
    Jphi = -Jx * jnp.sin(zeta) + Jy * jnp.cos(zeta)

    I_num = jnp.sum(Jphi * w) / (2.0 * jnp.pi * R0)

    assert jnp.allclose(I_num, I, rtol=1e-3)
