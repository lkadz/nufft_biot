import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from nufft_biot.forward import forward_B
from nufft_biot.types import BoxParams
from nufft_biot.utils import PeriodicFieldInterpolator, trace_field_line_rk4
from nufft_biot.current_models import torus_volume_current
from nufft_biot.embedding import embed_geometry_in_box

jax.config.update("jax_enable_x64", True)

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
I = 1e5

N_rho, N_theta, N_zeta = 64, 32, 64

box = BoxParams(
    Lx=10.0,
    Ly=10.0,
    Lz=10.0,
    Nx=256,
    Ny=256,
    Nz=256,
)

x_geom = jnp.array([major_radius])

X, Y, Z, Jx, Jy, Jz, w = torus_volume_current(
    I=I,
    major_radius=major_radius,
    minor_radius=minor_radius,
    N_rho=N_rho,
    N_theta=N_theta,
    N_zeta=N_zeta,
    use_nonuniform_grid=False,
)

Bx, By, Bz, center = forward_B(
    x_geom,
    box,
    I=I,
    major_radius=major_radius,
    minor_radius=minor_radius,
)

Bx = np.asarray(Bx)
By = np.asarray(By)
Bz = np.asarray(Bz)
center = np.asarray(center)

interp = PeriodicFieldInterpolator(Bx, By, Bz, box)

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
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
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

node_R = np.sqrt(np.asarray(X)**2 + np.asarray(Y)**2)
node_Z = np.asarray(Z)
node_R, node_Z = np.broadcast_arrays(node_R, node_Z)

ax.scatter(
    node_R.flatten(), 
    node_Z.flatten(), 
    s=15, 
    c='black', 
    marker='+', 
    alpha=0.6, 
    label="Grid Nodes"
)

theta = np.linspace(0.0, 2.0 * np.pi, 300)
coil_R = major_radius + minor_radius * np.cos(theta)
coil_Z = minor_radius * np.sin(theta)

ax.plot(
    coil_R,
    coil_Z,
    color="magenta",
    linestyle="--",
    linewidth=2.5,
    label="Torus boundary",
)

ax.annotate(
    "",
    xy=(major_radius, 0.0),
    xytext=(0.0, 0.0),
    arrowprops=dict(arrowstyle="<->", lw=2, color="black"),
)
ax.text(
    major_radius / 2,
    -0.08,
    rf"$R_0 = {major_radius}$",
    ha="center",
    va="top",
    fontsize=11,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
)
ax.annotate(
    "",
    xy=(major_radius, minor_radius),
    xytext=(major_radius, 0.0),
    arrowprops=dict(arrowstyle="<->", lw=2, color="black"),
)
ax.text(
    major_radius + 0.04,
    minor_radius / 2,
    rf"$a = {minor_radius}$",
    ha="left",
    va="center",
    fontsize=11,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
)

ax.set_aspect("equal")
ax.set_xlim(0.6, 1.45)
ax.set_ylim(-0.55, 0.55)
ax.set_xlabel("R")
ax.set_ylabel("Z")
ax.set_title("Poloidal Flux Surfaces")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

print("\n--- Starting Poloidal Comparison ---")

n_targets = 100
scan_radius = minor_radius * 0.5  
theta_test = np.linspace(0, 2 * np.pi, n_targets)

R_circle = major_radius + scan_radius * np.cos(theta_test)
Z_circle = scan_radius * np.sin(theta_test)

target_points = np.zeros((n_targets, 3))
target_points[:, 0] = R_circle + center[0]
target_points[:, 1] = 0.0 + center[1]
target_points[:, 2] = Z_circle + center[2]

B_nufft_list = interp(target_points) 
B_nufft = np.stack(B_nufft_list, axis=1)

Xb, Yb, Zb, _ = embed_geometry_in_box(X, Y, Z, box)
grid_epsilon = minor_radius / N_rho 

B_direct = biot_savart_direct(
    jnp.array(target_points),
    Xb.flatten(), Yb.flatten(), Zb.flatten(),
    Jx.flatten(), Jy.flatten(), Jz.flatten(),
    w.flatten(),
    epsilon=grid_epsilon
)
B_direct = np.asarray(B_direct)

B_mag_direct = np.linalg.norm(B_direct, axis=1)
B_mag_nufft = np.linalg.norm(B_nufft, axis=1)

diff = B_nufft - B_direct
error_vec = np.linalg.norm(diff, axis=1)
error_rel = error_vec / (B_mag_direct + 1e-9)

print(f"Max Relative Error: {np.max(error_rel):.2e}")
print(f"Mean Relative Error: {np.mean(error_rel):.2e}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(theta_test, B_mag_direct, 'k-', lw=2, label='Direct (Exact)')
ax.plot(theta_test, B_mag_nufft, 'r--', lw=2, label='NUFFT')
ax.set_ylabel('|B| [Tesla]')
ax.set_xlabel(r'Poloidal Angle $\theta$ [rad]')
ax.set_title(f'Field Magnitude at r = {scan_radius:.2f}m')
ax.legend()
ax.grid(True)

ax = axes[1, 0]
ax.semilogy(theta_test, error_rel, 'b.-')
ax.set_ylabel('Relative Error')
ax.set_xlabel(r'Poloidal Angle $\theta$ [rad]')
ax.set_title('Relative Error vs Angle')
ax.grid(True, which="both", ls="-", alpha=0.5)
ax.set_ylim(1e-5, 1.0) 

ax = axes[0, 1]
ax.plot(theta_test, B_direct[:, 2], 'k-', label=r'$B_z$ Direct')
ax.plot(theta_test, B_nufft[:, 2], 'r--', label=r'$B_z$ NUFFT')
ax.plot(theta_test, B_direct[:, 0], 'b-', label=r'$B_x$ Direct')
ax.plot(theta_test, B_nufft[:, 0], 'c--', label=r'$B_x$ NUFFT')
ax.plot(theta_test, B_direct[:, 1], 'g-', label=r'$B_y$ (Toroidal) Direct', alpha=0.6)
ax.plot(theta_test, B_nufft[:, 1], 'y--', label=r'$B_y$ (Toroidal) NUFFT', alpha=0.6)
ax.set_ylabel('Component Field [T]')
ax.set_xlabel(r'Poloidal Angle $\theta$ [rad]')
ax.set_title('B Components')
ax.legend()
ax.grid(True)

ax = axes[1, 1]
theta_draw = np.linspace(0, 2*np.pi, 100)
ax.plot(major_radius + minor_radius*np.cos(theta_draw), 
        minor_radius*np.sin(theta_draw), 'k--', label='Plasma Edge')
ax.plot(R_circle, Z_circle, 'r-', lw=2, label='Scan Path')
ax.plot(R_circle[0], Z_circle[0], 'ro', label='Start (theta=0)')
ax.set_aspect('equal')
ax.set_xlabel('R [m]')
ax.set_ylabel('Z [m]')
ax.set_title('Scan Geometry')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()