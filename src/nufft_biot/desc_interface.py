from __future__ import annotations
import numpy as np
import jax.numpy as jnp
try:
    from desc.grid import LinearGrid
except ImportError:
    LinearGrid = None

def desc_volume_current(
    eq, 
    *,
    L_grid: int | None = None, 
    M_grid: int | None = None, 
    N_grid: int | None = None
):
    if LinearGrid is None:
        raise ImportError("DESC is not installed. Please install it to use this feature.")

    L = L_grid if L_grid else eq.L_grid + 4
    M = M_grid if M_grid else eq.M_grid + 4
    N = N_grid if N_grid else (eq.N_grid * 2 if eq.N_grid else 32)

    grid = LinearGrid(L=L, M=M, N=N, sym=False, NFP=eq.NFP)

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
        X_list, Y_list, Z_list = [X], [Y], [Z]
        Jx_list, Jy_list, Jz_list = [Jx], [Jy], [Jz]
        w_list = [w]

        for k in range(1, eq.NFP):
            phi = 2.0 * jnp.pi * k / eq.NFP
            c, s = jnp.cos(phi), jnp.sin(phi)
            
            X_new = X * c - Y * s
            Y_new = X * s + Y * c
            Z_new = Z
            
            Jx_new = Jx * c - Jy * s
            Jy_new = Jx * s + Jy * c
            Jz_new = Jz

            X_list.append(X_new)
            Y_list.append(Y_new)
            Z_list.append(Z_new)
            Jx_list.append(Jx_new)
            Jy_list.append(Jy_new)
            Jz_list.append(Jz_new)
            w_list.append(w)
        
        X = jnp.concatenate(X_list)
        Y = jnp.concatenate(Y_list)
        Z = jnp.concatenate(Z_list)
        Jx = jnp.concatenate(Jx_list)
        Jy = jnp.concatenate(Jy_list)
        Jz = jnp.concatenate(Jz_list)
        w = jnp.concatenate(w_list)

    return X, Y, Z, Jx, Jy, Jz, w