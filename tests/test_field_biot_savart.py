import jax
import jax.numpy as jnp

from nufft_biot.current_models import torus_volume_current
from nufft_biot.embedding import embed_geometry_in_box
from nufft_biot.field import B_from_nodes_and_J
from nufft_biot.geometry import geom_from_x
from nufft_biot.types import BoxParams

jax.config.update("jax_enable_x64", True)


def direct_biot_savart_many(Robs, X, Y, Z, Jx, Jy, Jz, w):
    mu0 = 4.0 * jnp.pi * 1e-7
    src = jnp.stack([X, Y, Z], axis=1)
    J = jnp.stack([Jx, Jy, Jz], axis=1)

    def B_at_r(r):
        Rvec = r[None, :] - src
        r2 = jnp.sum(Rvec * Rvec, axis=1)
        r3 = jnp.where(r2 > 0.0, r2 * jnp.sqrt(r2), 1.0)
        cross = jnp.cross(J, Rvec)
        return mu0 / (4.0 * jnp.pi) * jnp.sum(cross * (w / r3)[:, None], axis=0)

    return jax.vmap(B_at_r)(Robs)


def test_volume_current_biot_savart_direction_consistency():
    R0 = 1.0
    a = 0.3
    I = 1.0

    box = BoxParams(
        Lx=8.0,
        Ly=8.0,
        Lz=8.0,
        Nx=64,
        Ny=64,
        Nz=64,
    )

    X, Y, Z, rho, theta, zeta, w = geom_from_x(
        jnp.zeros((1,)),
        N_rho=8,
        N_theta=16,
        N_zeta=32,
        major_radius=R0,
        minor_radius=a,
    )

    Jphi = I / (jnp.pi * a**2)
    Jx = -Jphi * jnp.sin(zeta)
    Jy = Jphi * jnp.cos(zeta)
    Jz = jnp.zeros_like(Jx)

    Xb, Yb, Zb, center = embed_geometry_in_box(X, Y, Z, box)

    Bx, By, Bz = B_from_nodes_and_J(
        Xb,
        Yb,
        Zb,
        Jx,
        Jy,
        Jz,
        w,
        box,
    )

    ix = jnp.round(Xb / box.Lx * box.Nx).astype(int) % box.Nx
    iy = jnp.round(Yb / box.Ly * box.Ny).astype(int) % box.Ny
    iz = jnp.round(Zb / box.Lz * box.Nz).astype(int) % box.Nz

    B_fft = jnp.stack(
        [Bx[ix, iy, iz], By[ix, iy, iz], Bz[ix, iy, iz]],
        axis=1,
    )

    Robs = jnp.stack([Xb, Yb, Zb], axis=1)
    B_bs = direct_biot_savart_many(
        Robs,
        Xb,
        Yb,
        Zb,
        Jx,
        Jy,
        Jz,
        w,
    )

    dot = jnp.sum(B_fft * B_bs, axis=1)
    denom = jnp.maximum(
        jnp.linalg.norm(B_fft, axis=1) * jnp.linalg.norm(B_bs, axis=1),
        1e-20,
    )

    cos_angle = dot / denom

    assert jnp.mean(cos_angle) > 0.85

def test_volume_current_biot_savart_magnitude_consistency():
    import jax.scipy.ndimage

    R0 = 1.0
    a = 0.3
    I = 1.0

    box = BoxParams(
        Lx=10.0,
        Ly=10.0,
        Lz=10.0,
        Nx=256,
        Ny=256,
        Nz=256,
    )

    X, Y, Z, rho, theta, zeta, w = geom_from_x(
        jnp.zeros((1,)),
        N_rho=10,
        N_theta=20,
        N_zeta=40,
        major_radius=R0,
        minor_radius=a,
    )

    Jphi = I / (jnp.pi * a**2)
    Jx = -Jphi * jnp.sin(zeta)
    Jy = Jphi * jnp.cos(zeta)
    Jz = jnp.zeros_like(Jx)

    Xb, Yb, Zb, center = embed_geometry_in_box(X, Y, Z, box)

    Bx, By, Bz = B_from_nodes_and_J(
        Xb, Yb, Zb, Jx, Jy, Jz, w, box,
    )

    u_x = (Xb / box.Lx) * box.Nx
    u_y = (Yb / box.Ly) * box.Ny
    u_z = (Zb / box.Lz) * box.Nz
    
    coords = jnp.stack([u_x, u_y, u_z], axis=0)

    Bx_interp = jax.scipy.ndimage.map_coordinates(Bx, coords, order=1, mode='wrap')
    By_interp = jax.scipy.ndimage.map_coordinates(By, coords, order=1, mode='wrap')
    Bz_interp = jax.scipy.ndimage.map_coordinates(Bz, coords, order=1, mode='wrap')

    B_fft = jnp.stack([Bx_interp, By_interp, Bz_interp], axis=1)

    Robs = jnp.stack([Xb, Yb, Zb], axis=1)
    B_bs = direct_biot_savart_many(
        Robs, Xb, Yb, Zb, Jx, Jy, Jz, w,
    )

    diff = B_fft - B_bs
    l2_error = jnp.linalg.norm(diff) / jnp.linalg.norm(B_bs)
    
    print(f"L2 Relative Error: {l2_error:.4e}")

    assert l2_error < 0.17

    