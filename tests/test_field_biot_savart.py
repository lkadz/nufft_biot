import jax.numpy as jnp
from nufft_biot.types import BoxParams
from nufft_biot.forward import geom_and_J_from_x
from nufft_biot.field import B_from_nodes_and_J


def _B_biot_savart_points(X, Y, Z, Jx, Jy, Jz, w, r_obs):
    mu0 = 4.0 * jnp.pi * 1e-7
    xs = jnp.stack([X, Y, Z], axis=1)
    Js = jnp.stack([Jx, Jy, Jz], axis=1)

    def B_at_r(r):
        R = r[None, :] - xs
        r2 = jnp.sum(R * R, axis=1)
        r3 = jnp.power(r2, 1.5)
        inv_r3 = jnp.where(r2 > 0.0, 1.0 / r3, 0.0)
        contrib = jnp.cross(Js, R) * (w * inv_r3)[:, None]
        return (mu0 / (4.0 * jnp.pi)) * jnp.sum(contrib, axis=0)

    return jnp.stack([B_at_r(r_obs[i]) for i in range(r_obs.shape[0])], axis=0)


def _sample_grid_nearest(Bx, By, Bz, box, r_obs):
    Nx, Ny, Nz = box.Nx, box.Ny, box.Nz
    dx, dy, dz = box.Lx / Nx, box.Ly / Ny, box.Lz / Nz

    x = jnp.mod(r_obs[:, 0], box.Lx)
    y = jnp.mod(r_obs[:, 1], box.Ly)
    z = jnp.mod(r_obs[:, 2], box.Lz)

    ix = jnp.mod(jnp.rint(x / dx).astype(int), Nx)
    iy = jnp.mod(jnp.rint(y / dy).astype(int), Ny)
    iz = jnp.mod(jnp.rint(z / dz).astype(int), Nz)

    return jnp.stack([Bx[ix, iy, iz], By[ix, iy, iz], Bz[ix, iy, iz]], axis=1)


def test_field_large_box_matches_direct_biot_savart_near_center():
    x = jnp.array([0.0])

    I = 1.0
    R0 = 1.0
    a = 0.3

    box = BoxParams(Lx=20.0, Ly=20.0, Lz=20.0, Nx=48, Ny=48, Nz=48)

    Xn, Yn, Zn, Jx, Jy, Jz, w = geom_and_J_from_x(
        x,
        I=I,
        major_radius=R0,
        minor_radius=a,
        N_rho=8,
        N_theta=16,
        N_zeta=32,
    )

    Bx, By, Bz = B_from_nodes_and_J(Xn, Yn, Zn, Jx, Jy, Jz, w, box)

    r_obs = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.2],
            [0.3, 0.0, 0.2],
        ],
        dtype=jnp.float64,
    )

    B_bs = _B_biot_savart_points(Xn, Yn, Zn, Jx, Jy, Jz, w, r_obs)
    B_fft = _sample_grid_nearest(Bx, By, Bz, box, r_obs)

    err = jnp.linalg.norm(B_fft - B_bs, axis=1)
    ref = jnp.linalg.norm(B_bs, axis=1)

    rel = err / (ref + 1e-30)

    assert jnp.all(rel < 0.25)
