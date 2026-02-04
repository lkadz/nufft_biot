from __future__ import annotations
import numpy as np
import jax.numpy as jnp
try:
    from desc.grid import LinearGrid
except ImportError:
    LinearGrid = None

def _apply_nfp_symmetry(NFP, X, Y, Z, Vx, Vy, Vz, w):
    """Helper to replicate points and vectors based on toroidal symmetry (NFP)."""
    X_list, Y_list, Z_list = [X], [Y], [Z]
    Vx_list, Vy_list, Vz_list = [Vx], [Vy], [Vz]
    w_list = [w]

    for k in range(1, NFP):
        phi = 2.0 * jnp.pi * k / NFP
        c, s = jnp.cos(phi), jnp.sin(phi)
        
        X_new = X * c - Y * s
        Y_new = X * s + Y * c
        Z_new = Z
        
        Vx_new = Vx * c - Vy * s
        Vy_new = Vx * s + Vy * c
        Vz_new = Vz

        X_list.append(X_new)
        Y_list.append(Y_new)
        Z_list.append(Z_new)
        Vx_list.append(Vx_new)
        Vy_list.append(Vy_new)
        Vz_list.append(Vz_new)
        w_list.append(w)
    
    return (
        jnp.concatenate(X_list),
        jnp.concatenate(Y_list),
        jnp.concatenate(Z_list),
        jnp.concatenate(Vx_list),
        jnp.concatenate(Vy_list),
        jnp.concatenate(Vz_list),
        jnp.concatenate(w_list)
    )

def desc_volume_current(
    eq, 
    *,
    L_grid: int | None = None, 
    M_grid: int | None = None, 
    N_grid: int | None = None
):
    """
    Extracts plasma volume current density (J) and integration weights (dV)
    from a DESC equilibrium object.
    """
    if LinearGrid is None:
        raise ImportError("DESC is not installed. Please install it to use this feature.")

    L = L_grid if L_grid else eq.L_grid + 4
    M = M_grid if M_grid else eq.M_grid + 4
    N = N_grid if N_grid else (eq.N_grid * 2 if eq.N_grid else 32)

    grid = LinearGrid(L=L, M=M, N=N, sym=False, NFP=eq.NFP, axis=False)

    keys = ["J", "X", "Y", "Z", "sqrt(g)"]
    data = eq.compute(keys, grid=grid, basis="xyz")

    X = jnp.array(data["X"])
    Y = jnp.array(data["Y"])
    Z = jnp.array(data["Z"])

    Jx = jnp.array(data["J"][:, 0])
    Jy = jnp.array(data["J"][:, 1])
    Jz = jnp.array(data["J"][:, 2])

    sqrt_g = jnp.array(data["sqrt(g)"])
    grid_weights = jnp.array(grid.weights)
    w = sqrt_g * grid_weights

    if eq.NFP > 1:
        X, Y, Z, Jx, Jy, Jz, w = _apply_nfp_symmetry(
            eq.NFP, X, Y, Z, Jx, Jy, Jz, w
        )

    return X, Y, Z, Jx, Jy, Jz, w


def desc_surface_current(
    field, 
    surface,
    *,
    M_grid: int = 120, 
    N_grid: int = 120
):
    """
    Extracts surface current density (K) and integration weights (dA)
    from a DESC FourierCurrentPotentialField and a corresponding surface.

    Parameters
    ----------
    field : FourierCurrentPotentialField
        The DESC field object containing the current potential phi.
    surface : Surface
        The DESC surface object (e.g. ConstantOffsetSurface) where the current lies.
    M_grid : int
        Poloidal grid resolution.
    N_grid : int
        Toroidal grid resolution.

    Returns
    -------
    X, Y, Z : jnp.ndarray
        Cartesian coordinates of the surface points.
    Kx, Ky, Kz : jnp.ndarray
        Cartesian components of the surface current density K.
    w : jnp.ndarray
        Integration weights (Area elements dA) for Biot-Savart integration.
    """
    if LinearGrid is None:
        raise ImportError("DESC is not installed.")

    grid = LinearGrid(M=M_grid, N=N_grid, NFP=surface.NFP, sym=False)
    
    keys = ["X", "Y", "Z", "K", "|e_theta x e_zeta|"]
    
    data = field.compute(keys, grid=grid, basis="xyz")
    
    X = jnp.array(data["X"])
    Y = jnp.array(data["Y"])
    Z = jnp.array(data["Z"])
    
    K_vec = jnp.array(data["K"])
    Kx = K_vec[:, 0]
    Ky = K_vec[:, 1]
    Kz = K_vec[:, 2]

    jacobian_surf = jnp.array(data["|e_theta x e_zeta|"])
    grid_weights = jnp.array(grid.weights)
    
    w = jacobian_surf * grid_weights

    if surface.NFP > 1:
        X, Y, Z, Kx, Ky, Kz, w = _apply_nfp_symmetry(
            surface.NFP, X, Y, Z, Kx, Ky, Kz, w
        )

    return X, Y, Z, Kx, Ky, Kz, w