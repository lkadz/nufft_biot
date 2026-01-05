from __future__ import annotations
import jax.numpy as jnp
from jax_finufft import nufft1

from .types import BoxParams


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
    mu0 = 4.0 * jnp.pi * 1e-7

    Nx, Ny, Nz = box.Nx, box.Ny, box.Nz
    shape = (Nx, Ny, Nz)

    tx = 2.0 * jnp.pi * X / box.Lx
    ty = 2.0 * jnp.pi * Y / box.Ly
    tz = 2.0 * jnp.pi * Z / box.Lz

    c_x = (Jx * w).astype(jnp.complex128)
    c_y = (Jy * w).astype(jnp.complex128)
    c_z = (Jz * w).astype(jnp.complex128)

    Jx_hat = nufft1(shape, c_x, tx, ty, tz, eps=eps)
    Jy_hat = nufft1(shape, c_y, tx, ty, tz, eps=eps)
    Jz_hat = nufft1(shape, c_z, tx, ty, tz, eps=eps)

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

    Bx_hat = jnp.where(mask, -1j * mu0 * kxJx / K2_safe, 0.0)
    By_hat = jnp.where(mask, -1j * mu0 * kxJy / K2_safe, 0.0)
    Bz_hat = jnp.where(mask, -1j * mu0 * kxJz / K2_safe, 0.0)

    Bx = jnp.fft.ifftn(Bx_hat).real
    By = jnp.fft.ifftn(By_hat).real
    Bz = jnp.fft.ifftn(Bz_hat).real

    return Bx, By, Bz

