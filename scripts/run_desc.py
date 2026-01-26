import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from desc.examples import get

from nufft_biot.desc_interface import desc_volume_current
from nufft_biot.embedding import make_optimal_box, embed_geometry_in_box
from nufft_biot.field import compute_B_hat, eval_B
from nufft_biot.utils import PeriodicFieldInterpolator, trace_field_line_rk4
from nufft_biot.utils.fast_tracing import trace_field_line_jax

jax.config.update("jax_enable_x64", True)

# -----------------------------------------------------------------------------
# 1. Reference Implementation (Direct Biot-Savart)
# -----------------------------------------------------------------------------
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
):
    mu0 = 4.0 * jnp.pi * 1e-7
    source_points = jnp.stack([source_X, source_Y, source_Z], axis=1)
    J_source = jnp.stack([Jx, Jy, Jz], axis=1)
    
    epsilon = 1e-10

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

# -----------------------------------------------------------------------------
# 2. Main Script
# -----------------------------------------------------------------------------
print("--- Loading DESC equilibrium (DSHAPE) ---")
eq = get("DSHAPE")

print("--- Extracting current density from DESC ---")
X_p, Y_p, Z_p, Jx_p, Jy_p, Jz_p, w_p = desc_volume_current(eq, N_grid=64)
print(f"    Plasma source points: {len(X_p)}")

print("--- Auto-sizing simulation box (Plasma Only) ---")
box = make_optimal_box(X_p, Y_p, Z_p, n_cells=180, padding=2.5)
print(f"    Box Length: {box.Lx:.2f}m, Grid: {box.Nx}^3")

R_source_p = np.sqrt(X_p**2 + Y_p**2)
Z_source_p = Z_p

R_c = (R_source_p.min() + R_source_p.max()) / 2.0
Z_c = (Z_source_p.min() + Z_source_p.max()) / 2.0

dist_from_center = np.sqrt((R_source_p - R_c)**2 + (Z_source_p - Z_c)**2)
a_minor = np.max(dist_from_center)

scan_radius = a_minor * 1.2
print(f"    Plasma Minor Radius: {a_minor:.2f} m")
print(f"    Scan Loop Radius:    {scan_radius:.2f} m")

R_cutoff = box.Lx / 2.0
if scan_radius >= R_cutoff:
    print("    WARNING: Scan radius exceeds box limits!")
else:
    print(f"    Safety Margin:       {scan_radius:.2f} m < {R_cutoff:.2f} m")

n_ghost = 5000
margin = 1.0 

x_min, x_max = X_p.min(), X_p.max()
y_min, y_max = Y_p.min(), Y_p.max()
z_min, z_max = Z_p.min(), Z_p.max()

X_ghost = np.random.uniform(x_min - margin, x_max + margin, n_ghost)
Y_ghost = np.random.uniform(y_min - margin, y_max + margin, n_ghost)
Z_ghost = np.random.uniform(z_min - margin, z_max + margin, n_ghost)

Jx_ghost = jnp.zeros(n_ghost)
Jy_ghost = jnp.zeros(n_ghost)
Jz_ghost = jnp.zeros(n_ghost)
w_ghost = jnp.zeros(n_ghost)

X = jnp.concatenate([X_p, jnp.array(X_ghost)])
Y = jnp.concatenate([Y_p, jnp.array(Y_ghost)])
Z = jnp.concatenate([Z_p, jnp.array(Z_ghost)])
Jx = jnp.concatenate([Jx_p, Jx_ghost])
Jy = jnp.concatenate([Jy_p, Jy_ghost])
Jz = jnp.concatenate([Jz_p, Jz_ghost])
w = jnp.concatenate([w_p, w_ghost])

print(f"    Added {n_ghost} ghost points.")
print(f"    Total Source points: {len(X)}")

Xb, Yb, Zb, shift = embed_geometry_in_box(X, Y, Z, box)

print("--- Computing B field modes via NUFFT ---")
Bx_hat, By_hat, Bz_hat = compute_B_hat(Xb, Yb, Zb, Jx, Jy, Jz, w, box)

print("\n--- Running Accuracy Validation ---")

n_scan = 100
theta_scan = np.linspace(0, 2 * np.pi, n_scan)
scan_R = R_c + scan_radius * np.cos(theta_scan)
scan_Z = Z_c + scan_radius * np.sin(theta_scan)
scan_Y = np.zeros_like(scan_R)

targets_local = np.stack([scan_R, scan_Y, scan_Z], axis=1)
targets_box = targets_local - np.array(shift)

print("    Computing Direct Biot-Savart (Reference)...")
B_direct = biot_savart_direct(
    jnp.array(targets_box), 
    Xb, Yb, Zb, Jx, Jy, Jz, w
)

print("    Computing NUFFT Spectral Readout...")
bx_n, by_n, bz_n = eval_B(
    Bx_hat, By_hat, Bz_hat, jnp.array(targets_box), box
)
B_nufft = np.stack([bx_n, by_n, bz_n], axis=1)

diff = B_nufft - B_direct
err = np.linalg.norm(diff, axis=1)
mag = np.linalg.norm(B_direct, axis=1)
rel_err = err / (mag.max() + 1e-12)

print(f"    Max Absolute Error: {np.max(err):.4e} T")
print(f"    Max Relative Error: {np.max(rel_err):.4e}")

print("\n--- Generating Field Line Trace (JAX Accelerated) ---")

Bx_grid = jnp.fft.ifftn(jnp.fft.ifftshift(Bx_hat) * box.N_total).real
By_grid = jnp.fft.ifftn(jnp.fft.ifftshift(By_hat) * box.N_total).real
Bz_grid = jnp.fft.ifftn(jnp.fft.ifftshift(Bz_hat) * box.N_total).real

seeds = []
step_sizes = []

for r in np.linspace(R_c + 0.4, R_c + a_minor * 0.9, 8):
    seed = np.array([r, 0.0, 0.0]) - np.array(shift)
    seeds.append(jnp.array(seed))
    
    radius = abs(r - R_c)
    ds_adaptive = np.clip(radius * 0.2, 0.005, 0.05)
    step_sizes.append(ds_adaptive)

field_lines = []

print("    JIT Compiling and Tracing...")

for seed, ds in zip(seeds, step_sizes):
    line = trace_field_line_jax(
        Bx_grid, By_grid, Bz_grid, 
        seed, 
        box.Lx, box.Ly, box.Lz,
        ds=ds, 
        n_steps=10000,
        order=3
    )
    field_lines.append(np.array(line))

print("    Done.")

# -----------------------------------------------------------------------------
# 3. Plotting
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1)
for line in field_lines:
    R_line = np.sqrt(line[:,0]**2 + line[:,1]**2)
    Z_line = line[:,2]
    ax1.plot(R_line, Z_line)

ax1.scatter(np.sqrt(X_p**2 + Y_p**2)[::20], Z_p[::20], s=0.1, c='k', alpha=0.1, label="Current Source")

ax1.plot(scan_R, scan_Z, 'r--', lw=2, label="Validation Loop")

ax1.axis('equal')
ax1.set_xlabel("R [m]")
ax1.set_ylabel("Z [m]")
ax1.set_title("Field Line Tracing (NUFFT)")
ax1.grid(True)
ax1.legend(loc='upper right')

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(theta_scan, np.linalg.norm(B_direct, axis=1), 'k-', lw=2, label="Direct Biot-Savart")
ax2.plot(theta_scan, np.linalg.norm(B_nufft, axis=1), 'r--', lw=2, label="NUFFT (Spectral)")

ax2_right = ax2.twinx()
ax2_right.semilogy(theta_scan, rel_err, 'b:', label="Relative Error")
ax2_right.set_ylabel("Relative Error", color='b')
ax2_right.tick_params(axis='y', labelcolor='b')

ax2.set_xlabel("Poloidal Angle (radians)")
ax2.set_ylabel("|B| [Tesla]")
ax2.set_title(f"Validation on Loop (r = {scan_radius:.2f} m)")
ax2.set_xlim(0, 2*np.pi)
ax2.grid(True)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_right.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

plt.tight_layout()
plt.show()