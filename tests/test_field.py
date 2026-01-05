import jax.numpy as jnp

from nufft_biot.forward import forward_B
from nufft_biot.types import BoxParams


def make_box(L=6.0, N=32):
    return BoxParams(
        Lx=L,
        Ly=L,
        Lz=L,
        Nx=N,
        Ny=N,
        Nz=N,
    )


def test_zero_current_gives_zero_field():
    box = make_box()
    x = jnp.array([0.0])

    Bx, By, Bz, _ = forward_B(
        x,
        box,
        I=0.0,
        major_radius=1.0,
        minor_radius=0.3,
        N_rho=6,
        N_theta=12,
        N_zeta=24,
    )

    assert jnp.allclose(Bx, 0.0)
    assert jnp.allclose(By, 0.0)
    assert jnp.allclose(Bz, 0.0)


def test_field_midplane_symmetry():
    box = make_box()
    x = jnp.array([0.0])

    Bx, By, Bz, _ = forward_B(
        x,
        box,
        I=1.0,
        major_radius=1.2,
        minor_radius=0.3,
        N_rho=6,
        N_theta=12,
        N_zeta=24,
    )

    Nz = box.Nz
    mid = Nz // 2

    assert jnp.allclose(Bx[:, :, mid], Bx[:, :, -mid], atol=1e-6)
    assert jnp.allclose(By[:, :, mid], By[:, :, -mid], atol=1e-6)
    assert jnp.allclose(Bz[:, :, mid], -Bz[:, :, -mid], atol=1e-6)


def test_field_linearity():
    box = make_box()
    x = jnp.array([0.0])

    Bx1, By1, Bz1, _ = forward_B(
        x,
        box,
        I=1.0,
        major_radius=1.1,
        minor_radius=0.25,
        N_rho=6,
        N_theta=12,
        N_zeta=24,
    )

    Bx2, By2, Bz2, _ = forward_B(
        x,
        box,
        I=2.0,
        major_radius=1.1,
        minor_radius=0.25,
        N_rho=6,
        N_theta=12,
        N_zeta=24,
    )

    assert jnp.allclose(Bx2, 2.0 * Bx1, rtol=1e-12)
    assert jnp.allclose(By2, 2.0 * By1, rtol=1e-12)
    assert jnp.allclose(Bz2, 2.0 * Bz1, rtol=1e-12)


def test_divergence_free_field():
    box = make_box()
    x = jnp.array([0.0])

    Bx, By, Bz, _ = forward_B(
        x,
        box,
        I=1.0,
        major_radius=1.1,
        minor_radius=0.25,
        N_rho=6,
        N_theta=12,
        N_zeta=24,
    )

    dx = box.Lx / box.Nx
    dy = box.Ly / box.Ny
    dz = box.Lz / box.Nz

    divB = (
        jnp.gradient(Bx, dx, axis=0)
        + jnp.gradient(By, dy, axis=1)
        + jnp.gradient(Bz, dz, axis=2)
    )

    total_div = jnp.sum(divB) * dx * dy * dz

    assert jnp.allclose(total_div, 0.0, atol=1e-10)
