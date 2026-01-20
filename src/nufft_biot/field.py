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

    tx = 2.0 * jnp.pi * X / box.Lx
    ty = 2.0 * jnp.pi * Y / box.Ly
    tz = 2.0 * jnp.pi * Z / box.Lz

    c_x = (Jx * w).astype(jnp.complex128)
    c_y = (Jy * w).astype(jnp.complex128)
    c_z = (Jz * w).astype(jnp.complex128)

    Jx_hat = nufft1(shape, c_x, tx, ty, tz, eps=eps, iflag=-1)
    Jy_hat = nufft1(shape, c_y, tx, ty, tz, eps=eps, iflag=-1)
    Jz_hat = nufft1(shape, c_z, tx, ty, tz, eps=eps, iflag=-1)

    kx = 2.0 * jnp.pi * jnp.fft.fftshift(jnp.fft.fftfreq(Nx, d=box.Lx / Nx))
    ky = 2.0 * jnp.pi * jnp.fft.fftshift(jnp.fft.fftfreq(Ny, d=box.Ly / Ny))
    kz = 2.0 * jnp.pi * jnp.fft.fftshift(jnp.fft.fftfreq(Nz, d=box.Lz / Nz))

    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")
    K_mag = jnp.sqrt(KX**2 + KY**2 + KZ**2)

    R_cut = min(box.Lx, box.Ly, box.Lz) / 2.0

    mask = K_mag > 0.0
    K2_safe = jnp.where(mask, K_mag**2, 1.0)
    trunc_factor = (1.0 - jnp.cos(K_mag * R_cut)) / K2_safe

    dc_term = 0.5 * R_cut**2

    greens_func = jnp.where(mask, trunc_factor, dc_term)

    k_dot_J = KX * Jx_hat + KY * Jy_hat + KZ * Jz_hat
    proj_factor = k_dot_J / K2_safe

    Jx_proj = jnp.where(mask, Jx_hat - KX * proj_factor, Jx_hat)
    Jy_proj = jnp.where(mask, Jy_hat - KY * proj_factor, Jy_hat)
    Jz_proj = jnp.where(mask, Jz_hat - KZ * proj_factor, Jz_hat)

    kxJx = KY * Jz_proj - KZ * Jy_proj
    kxJy = KZ * Jx_proj - KX * Jz_proj
    kxJz = KX * Jy_proj - KY * Jx_proj

    factor = 1j * mu0 * greens_func

    Bx_hat = factor * kxJx
    By_hat = factor * kxJy
    Bz_hat = factor * kxJz

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
    tx = 2.0 * jnp.pi * (target_pos[:, 0] % box.Lx) / box.Lx
    ty = 2.0 * jnp.pi * (target_pos[:, 1] % box.Ly) / box.Ly
    tz = 2.0 * jnp.pi * (target_pos[:, 2] % box.Lz) / box.Lz

    shape = Bx_hat.shape
    N_total = shape[0] * shape[1] * shape[2]
    scale = 1.0 / N_total

    Bx = nufft2(Bx_hat, tx, ty, tz, eps=eps, iflag=1).real * scale
    By = nufft2(By_hat, tx, ty, tz, eps=eps, iflag=1).real * scale
    Bz = nufft2(Bz_hat, tx, ty, tz, eps=eps, iflag=1).real * scale

    return Bx, By, Bz
