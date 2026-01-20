import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from nufft_biot.forward import forward_B
from nufft_biot.types import BoxParams
from nufft_biot.utils import PeriodicFieldInterpolator, trace_field_line_rk4
from nufft_biot.current_models import torus_volume_current
from nufft_biot.embedding import embed_geometry_in_box
from nufft_biot.geometry import geom_from_x
from nufft_biot.field import compute_B_modes, eval_B_at_targets

jax.config.update("jax_enable_x64", True)


def ellipk(m):
    a0 = 1.0
    b0 = jnp.sqrt(1.0 - m)
    for _ in range(6):
        a_new = (a0 + b0) / 2.0
        b_new = jnp.sqrt(a0 * b0)
        a0, b0 = a_new, b_new
    return jnp.pi / (2.0 * a0)


def ellipe(m):
    a = 1.0
    b = jnp.sqrt(1.0 - m)
    c = jnp.sqrt(m)
    ls = []
    for i in range(6):
        ls.append(c**2 * (2.0**i))
        a_new = (a + b) / 2.0
        b_new = jnp.sqrt(a * b)
        c_new = (a - b) / 2.0
        a, b, c = a_new, b_new, c_new
    sum_term = jnp.sum(jnp.stack(ls), axis=0)
    K = jnp.pi / (2.0 * a)
    return K * (1.0 - sum_term / 2.0)


@jax.jit
def biot_savart_elliptic_integral(
    target_points: jnp.ndarray,
    source_R: jnp.ndarray,
    source_Z: jnp.ndarray,
    I_source: jnp.ndarray,
    core_radius: float = 1e-3,
):
    mu0 = 4.0 * jnp.pi * 1e-7
    T_x, T_y, T_z = target_points[:, 0], target_points[:, 1], target_points[:, 2]
    r_t = jnp.sqrt(T_x**2 + T_y**2)
    phi_t = jnp.arctan2(T_y, T_x)
    z_t = T_z
    r_t_bc = r_t[:, None]
    z_t_bc = z_t[:, None]
    r_s = source_R[None, :]
    z_s = source_Z[None, :]
    I = I_source[None, :]
    dz = z_t_bc - z_s
    alpha_sq = (r_s - r_t_bc) ** 2 + dz**2 + core_radius**2
    beta_sq = (r_s + r_t_bc) ** 2 + dz**2
    k_sq = (4.0 * r_t_bc * r_s) / (beta_sq + 1e-20)
    k_sq = jnp.clip(k_sq, 0.0, 1.0 - 1e-12)
    K = ellipk(k_sq)
    E = ellipe(k_sq)
    denom = jnp.sqrt(beta_sq)
    B_rho_term = (
        (mu0 * I * dz)
        / (2.0 * jnp.pi * r_t_bc * denom + 1e-12)
        * (-K + E * ((r_s**2 + r_t_bc**2 + dz**2) / alpha_sq))
    )
    B_rho = jnp.where(r_t_bc < 1e-10, 0.0, B_rho_term)
    B_z_term = (
        (mu0 * I)
        / (2.0 * jnp.pi * denom)
        * (K + E * ((r_s**2 - r_t_bc**2 - dz**2) / alpha_sq))
    )
    B_rho_total = jnp.sum(B_rho, axis=1)
    B_z_total = jnp.sum(B_z_term, axis=1)
    Bx = B_rho_total * jnp.cos(phi_t)
    By = B_rho_total * jnp.sin(phi_t)
    Bz = B_z_total
    return jnp.stack([Bx, By, Bz], axis=1)


@jax.jit
def biot_savart_direct(
    target_points: jnp.ndarray,
    source_X: jnp.ndarray,
    source_Y: jnp.ndarray,
    source_Z: jnp.ndarray,
    Jx: jnp.ndarray,
    Jy: jnp.ndarray,
    Jz: jnp.ndarray,
    w: jnp.ndarray,
    epsilon: float,
):
    mu0 = 4.0 * jnp.pi * 1e-7
    source_points = jnp.stack([source_X, source_Y, source_Z], axis=1)
    J_source = jnp.stack([Jx, Jy, Jz], axis=1)

    def compute_single_point(r_target):
        r_vec = r_target - source_points
        dist_sq = jnp.sum(r_vec**2, axis=1)
        dist_soft = jnp.sqrt(dist_sq + epsilon**2)
        factor = w / (dist_soft**3)
        cross_x = J_source[:, 1] * r_vec[:, 2] - J_source[:, 2] * r_vec[:, 1]
        cross_y = J_source[:, 2] * r_vec[:, 0] - J_source[:, 0] * r_vec[:, 2]
        cross_z = J_source[:, 0] * r_vec[:, 1] - J_source[:, 1] * r_vec[:, 0]
        Bx = (mu0 / (4.0 * jnp.pi)) * jnp.sum(cross_x * factor)
        By = (mu0 / (4.0 * jnp.pi)) * jnp.sum(cross_y * factor)
        Bz = (mu0 / (4.0 * jnp.pi)) * jnp.sum(cross_z * factor)
        return jnp.array([Bx, By, Bz])

    return jax.vmap(compute_single_point)(target_points)


major_radius = 1.0
minor_radius = 0.3
I_total = 1e5

N_rho, N_theta, N_zeta = 128, 128, 128

box = BoxParams(
    Lx=8.0,
    Ly=8.0,
    Lz=8.0,
    Nx=256,
    Ny=256,
    Nz=256,
)

x_geom = jnp.array([major_radius])

X, Y, Z, Jx, Jy, Jz, w = torus_volume_current(
    I=I_total,
    major_radius=major_radius,
    minor_radius=minor_radius,
    N_rho=N_rho,
    N_theta=N_theta,
    N_zeta=N_zeta,
    use_nonuniform_grid=False,
)

Bx_nu, By_nu, Bz_nu, center = forward_B(
    x_geom,
    box,
    I=I_total,
    major_radius=major_radius,
    minor_radius=minor_radius,
)

Bx_nu = np.asarray(Bx_nu)
By_nu = np.asarray(By_nu)
Bz_nu = np.asarray(Bz_nu)
center = np.asarray(center)

interp = PeriodicFieldInterpolator(Bx_nu, By_nu, Bz_nu, box)

phi0 = 0.0
seeds = []
for r in np.linspace(0.2, 0.9 * minor_radius, 10):
    seed = np.array(
        [
            (major_radius + r) * np.cos(phi0),
            (major_radius + r) * np.sin(phi0),
            0.0,
        ],
        dtype=float,
    )
    seed += center
    seeds.append(seed)

field_lines = [
    trace_field_line_rk4(interp, seed, box, ds=0.01, n_steps=4000)
    for seed in seeds
]

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")
for fl in field_lines:
    flc = fl - center
    ax.plot(flc[:, 0], flc[:, 1], flc[:, 2])
ax.set_box_aspect([1, 1, 1])
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)
ax.set_title("3D Poloidal Flux Surfaces")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
ax = plt.gca()
for fl in field_lines:
    flc = fl - center
    fl_R = np.sqrt(flc[:, 0] ** 2 + flc[:, 1] ** 2)
    fl_Z = flc[:, 2]
    ax.plot(fl_R, fl_Z, linewidth=1.5, alpha=0.8)

theta = np.linspace(0.0, 2.0 * np.pi, 300)
coil_R = major_radius + minor_radius * np.cos(theta)
coil_Z = minor_radius * np.sin(theta)
ax.plot(coil_R, coil_Z, "m--", lw=2.5, label="Torus boundary")

ax.set_aspect("equal")
ax.set_xlim(0.6, 1.45)
ax.set_ylim(-0.55, 0.55)
ax.set_title("Poloidal Flux Surfaces")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

print("\n--- Starting Poloidal Comparison ---")

n_targets = 100
scan_radius = minor_radius * 2

theta_test = np.linspace(0, 2 * np.pi, n_targets)
R_circle = major_radius + scan_radius * np.cos(theta_test)
Z_circle = scan_radius * np.sin(theta_test)

target_points = np.zeros((n_targets, 3))
target_points[:, 0] = R_circle + center[0]
target_points[:, 1] = center[1]
target_points[:, 2] = Z_circle + center[2]

Xb, Yb, Zb, _ = embed_geometry_in_box(X, Y, Z, box)

Bx_hat, By_hat, Bz_hat = compute_B_modes(
    Xb, Yb, Zb, Jx, Jy, Jz, w, box, eps=1e-14
)

bx_spec, by_spec, bz_spec = eval_B_at_targets(
    Bx_hat, By_hat, Bz_hat, jnp.array(target_points), box, eps=1e-14
)

B_nufft = np.stack([bx_spec, by_spec, bz_spec], axis=1)

slice_idx = 0
shape = (N_rho, N_theta, N_zeta)
Xr = X.reshape(shape)
Yr = Y.reshape(shape)
Zr = Z.reshape(shape)
Jxr = Jx.reshape(shape)
Jyr = Jy.reshape(shape)
phi0_j = jnp.arctan2(Yr[0, 0, slice_idx], Xr[0, 0, slice_idx])
sinp, cosp = jnp.sin(phi0_j), jnp.cos(phi0_j)
Jphi_slice = -Jxr[:, :, slice_idx] * sinp + Jyr[:, :, slice_idx] * cosp

rho_nodes_1d = (jnp.arange(N_rho) + 0.5) / N_rho
rho_grid = rho_nodes_1d[:, None] * jnp.ones((1, N_theta), dtype=jnp.float64)
drho = 1.0 / N_rho
dtheta = 2.0 * jnp.pi / N_theta
dA = (minor_radius**2) * rho_grid * drho * dtheta
I_cells = Jphi_slice * dA

R_slice = jnp.sqrt(Xr[:, :, slice_idx] ** 2 + Yr[:, :, slice_idx] ** 2)
Z_slice = Zr[:, :, slice_idx]
R_flat = R_slice.reshape(-1)
Z_flat = Z_slice.reshape(-1)
I_flat = I_cells.reshape(-1)

dr_est = minor_radius / N_rho
core_radius_est = 0.5 * dr_est
target_points_local = target_points - center

B_elliptic = biot_savart_elliptic_integral(
    jnp.array(target_points_local),
    R_flat,
    Z_flat,
    I_flat,
    core_radius=core_radius_est,
)
B_elliptic = np.asarray(B_elliptic)

B_direct = biot_savart_direct(
    jnp.array(target_points),
    Xb.flatten(),
    Yb.flatten(),
    Zb.flatten(),
    Jx.flatten(),
    Jy.flatten(),
    Jz.flatten(),
    w.flatten(),
    epsilon=1e-10,
)
B_direct = np.asarray(B_direct)

B_mag_elliptic = np.linalg.norm(B_elliptic, axis=1)
B_mag_nufft = np.linalg.norm(B_nufft, axis=1)

diff_e = B_nufft - B_elliptic
error_vec_e = np.linalg.norm(diff_e, axis=1)
error_rel_e = error_vec_e / (B_mag_elliptic + 1e-9)

print(f"\n[Elliptic] Max Relative Error:  {np.max(error_rel_e):.2e}")
print(f"[Elliptic] Mean Relative Error: {np.mean(error_rel_e):.2e}")

B_mag_direct = np.linalg.norm(B_direct, axis=1)
diff_d = B_nufft - B_direct
error_vec_d = np.linalg.norm(diff_d, axis=1)
error_rel_d = error_vec_d / (B_mag_direct + 1e-9)

print(f"\n[Direct]   Max Relative Error:  {np.max(error_rel_d):.2e}")
print(f"[Direct]   Mean Relative Error: {np.mean(error_rel_d):.2e}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(theta_test, B_mag_elliptic, "k-", lw=2, label="Elliptic Integral")
ax.plot(theta_test, B_mag_direct, "b-", lw=2, label="Direct Biot–Savart")
ax.plot(theta_test, B_mag_nufft, "r--", lw=2, label="NUFFT (Spectral)")
ax.set_ylabel("|B| [Tesla]")
ax.set_title(f"Field Magnitude at r = {scan_radius:.2f} m")
ax.legend()
ax.grid(True)

ax = axes[1, 0]
ax.semilogy(theta_test, error_rel_e, "k.-", label="NUFFT vs Elliptic")
ax.semilogy(theta_test, error_rel_d, "b.-", label="NUFFT vs Direct")
ax.set_ylabel("Relative Error")
ax.set_title("Relative Error vs Angle")
ax.grid(True, which="both", ls="-", alpha=0.5)
ax.set_ylim(1e-12, 1.0)
ax.legend()

ax = axes[0, 1]
ax.plot(theta_test, B_elliptic[:, 2], "k-", label=r"$B_z$ Elliptic")
ax.plot(theta_test, B_nufft[:, 2], "r--", label=r"$B_z$ NUFFT")
ax.plot(theta_test, B_elliptic[:, 0], "b-", label=r"$B_x$ Elliptic")
ax.plot(theta_test, B_nufft[:, 0], "c--", label=r"$B_x$ NUFFT")
ax.plot(theta_test, B_direct[:, 2], "k:", lw=1.5, label=r"$B_z$ Direct")
ax.set_title("B Components")
ax.legend()
ax.grid(True)

ax = axes[1, 1]
theta_draw = np.linspace(0, 2 * np.pi, 100)
ax.plot(
    major_radius + minor_radius * np.cos(theta_draw),
    minor_radius * np.sin(theta_draw),
    "k--",
    label="Plasma Edge",
)
ax.plot(R_circle, Z_circle, "r-", lw=2, label="Scan Path")
ax.set_aspect("equal")
ax.set_title("Scan Geometry")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

I_slice = jnp.sum(Jphi_slice * dA)
print("\nNormalization Check:")
print("I_slice =", float(I_slice), "target =", I_total, "ratio =", float(I_slice / I_total))
