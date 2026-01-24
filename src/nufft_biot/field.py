from __future__ import annotations
import jax.numpy as jnp
from jax_finufft import nufft1, nufft2

from .types import BoxParams


def compute_B_modes(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    Z: jnp.ndarray,
    Jx: jnp.ndarray,
    Jy: jnp.ndarray,
    Jz: jnp.ndarray,
    w: jnp.ndarray,
    box: BoxParams,
    *,
    eps: float = 1e-6,
):
    mu0 = 4.0 * jnp.pi * 1e-7

    Nx, Ny, Nz = box.Nx, box.Ny, box.Nz
    shape = (Nx, Ny, Nz)

    tx = 2.0 * jnp.pi * (X / box.Lx) - jnp.pi
    ty = 2.0 * jnp.pi * (Y / box.Ly) - jnp.pi
    tz = 2.0 * jnp.pi * (Z / box.Lz) - jnp.pi

    c_x = (Jx * w).astype(jnp.complex128)
    c_y = (Jy * w).astype(jnp.complex128)
    c_z = (Jz * w).astype(jnp.complex128)

    Jx_hat = nufft1(shape, c_x, tx, ty, tz, eps=eps, iflag=-1)
    Jy_hat = nufft1(shape, c_y, tx, ty, tz, eps=eps, iflag=-1)
    Jz_hat = nufft1(shape, c_z, tx, ty, tz, eps=eps, iflag=-1)

    k_vec_x = jnp.arange(-Nx // 2, Nx // 2)
    k_vec_y = jnp.arange(-Ny // 2, Ny // 2)
    k_vec_z = jnp.arange(-Nz // 2, Nz // 2)

    phase_x = jnp.where(k_vec_x % 2 == 0, 1.0, -1.0)
    phase_y = jnp.where(k_vec_y % 2 == 0, 1.0, -1.0)
    phase_z = jnp.where(k_vec_z % 2 == 0, 1.0, -1.0)

    phase_grid = (
        phase_x[:, None, None]
        * phase_y[None, :, None]
        * phase_z[None, None, :]
    )

    Jx_hat *= phase_grid
    Jy_hat *= phase_grid
    Jz_hat *= phase_grid

    Jx_hat = jnp.fft.ifftshift(Jx_hat)
    Jy_hat = jnp.fft.ifftshift(Jy_hat)
    Jz_hat = jnp.fft.ifftshift(Jz_hat)

    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(Nx, d=box.Lx / Nx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(Ny, d=box.Ly / Ny)
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(Nz, d=box.Lz / Nz)

    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2

    mask = K2 > 0.0
    K2_safe = jnp.where(mask, K2, 1.0)

    k_dot_J = KX * Jx_hat + KY * Jy_hat + KZ * Jz_hat
    Jx_hat = jnp.where(mask, Jx_hat - KX * k_dot_J / K2_safe, 0.0)
    Jy_hat = jnp.where(mask, Jy_hat - KY * k_dot_J / K2_safe, 0.0)
    Jz_hat = jnp.where(mask, Jz_hat - KZ * k_dot_J / K2_safe, 0.0)

    kxJx = KY * Jz_hat - KZ * Jy_hat
    kxJy = KZ * Jx_hat - KX * Jz_hat
    kxJz = KX * Jy_hat - KY * Jx_hat

    R_cut = min(box.Lx, box.Ly, box.Lz) / 2.0
    k_mag = jnp.sqrt(K2_safe)
    trunc_factor = 1.0 - jnp.cos(k_mag * R_cut)

    Bx_hat = jnp.where(mask, (1j * mu0 * kxJx / K2_safe) * trunc_factor, 0.0)
    By_hat = jnp.where(mask, (1j * mu0 * kxJy / K2_safe) * trunc_factor, 0.0)
    Bz_hat = jnp.where(mask, (1j * mu0 * kxJz / K2_safe) * trunc_factor, 0.0)

    volume_factor = (Nx * Ny * Nz) / (box.Lx * box.Ly * box.Lz)
    Bx_hat *= volume_factor
    By_hat *= volume_factor
    Bz_hat *= volume_factor

    return Bx_hat, By_hat, Bz_hat


def B_from_nodes_and_J(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    Z: jnp.ndarray,
    Jx: jnp.ndarray,
    Jy: jnp.ndarray,
    Jz: jnp.ndarray,
    w: jnp.ndarray,
    box: BoxParams,
    *,
    eps: float = 1e-6,
):
    Bx_hat, By_hat, Bz_hat = compute_B_modes(X, Y, Z, Jx, Jy, Jz, w, box, eps=eps)

    Bx = jnp.fft.ifftn(jnp.fft.ifftshift(Bx_hat)).real
    By = jnp.fft.ifftn(jnp.fft.ifftshift(By_hat)).real
    Bz = jnp.fft.ifftn(jnp.fft.ifftshift(Bz_hat)).real

    return Bx, By, Bz


def eval_B_at_targets(
    Bx_hat: jnp.ndarray,
    By_hat: jnp.ndarray,
    Bz_hat: jnp.ndarray,
    target_pos: jnp.ndarray,
    box: BoxParams,
    eps: float = 1e-6,
):
    tx_phys = target_pos[:, 0] % box.Lx
    ty_phys = target_pos[:, 1] % box.Ly
    tz_phys = target_pos[:, 2] % box.Lz

    tx = 2.0 * jnp.pi * (tx_phys / box.Lx) - jnp.pi
    ty = 2.0 * jnp.pi * (ty_phys / box.Ly) - jnp.pi
    tz = 2.0 * jnp.pi * (tz_phys / box.Lz) - jnp.pi

    Bx_cen = jnp.fft.fftshift(Bx_hat)
    By_cen = jnp.fft.fftshift(By_hat)
    Bz_cen = jnp.fft.fftshift(Bz_hat)

    shape = Bx_hat.shape
    Nx, Ny, Nz = shape

    k_vec_x = jnp.arange(-Nx // 2, Nx // 2)
    k_vec_y = jnp.arange(-Ny // 2, Ny // 2)
    k_vec_z = jnp.arange(-Nz // 2, Nz // 2)

    phase_x = jnp.where(k_vec_x % 2 == 0, 1.0, -1.0)
    phase_y = jnp.where(k_vec_y % 2 == 0, 1.0, -1.0)
    phase_z = jnp.where(k_vec_z % 2 == 0, 1.0, -1.0)

    phase_grid = (
        phase_x[:, None, None]
        * phase_y[None, :, None]
        * phase_z[None, None, :]
    )

    Bx_cen *= phase_grid
    By_cen *= phase_grid
    Bz_cen *= phase_grid

    N_total = shape[0] * shape[1] * shape[2]
    scale = 1.0 / N_total

    Bx = nufft2(Bx_cen, tx, ty, tz, eps=eps, iflag=1).real * scale
    By = nufft2(By_cen, tx, ty, tz, eps=eps, iflag=1).real * scale
    Bz = nufft2(Bz_cen, tx, ty, tz, eps=eps, iflag=1).real * scale

    return Bx, By, Bz
