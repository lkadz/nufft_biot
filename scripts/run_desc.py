import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from desc.examples import get

from nufft_biot.desc_interface import desc_volume_current
from nufft_biot.embedding import make_optimal_box, embed_geometry_in_box
from nufft_biot.field import compute_B_modes, eval_B_at_targets
from nufft_biot.utils import PeriodicFieldInterpolator, trace_field_line_rk4

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
    """
    Computes B at target_points using naive summing of (J x r) / r^3.
    """
    mu0 = 4.0 * jnp.pi * 1e-7
    source_points = jnp.stack([source_X, source_Y, source_Z], axis=1)
    J_source = jnp.stack([Jx, Jy, Jz], axis=1)
    
    # Epsilon to prevent division by zero if target overlaps source exactly
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

# A. Load DESC Equilibrium
print("--- Loading DESC equilibrium (DSHAPE) ---")
eq = get("DSHAPE")

# B. Extract Geometry
print("--- Extracting current density from DESC ---")
# N_grid=64 ensures smooth toroidal curvature
X, Y, Z, Jx, Jy, Jz, w = desc_volume_current(eq, N_grid=64)
print(f"    Source points: {len(X)}")

# C. Setup Box
print("--- Auto-sizing simulation box ---")
box = make_optimal_box(X, Y, Z, n_cells=300, padding=2.5)
print(f"    Box Length: {box.Lx:.2f}m, Grid: {box.Nx}^3")

# D. Embed Geometry
Xb, Yb, Zb, center = embed_geometry_in_box(X, Y, Z, box)

# E. Compute Physics (Spectral Modes)
print("--- Computing B field modes via NUFFT ---")
Bx_hat, By_hat, Bz_hat = compute_B_modes(Xb, Yb, Zb, Jx, Jy, Jz, w, box)

# F. Comparison: Direct vs NUFFT (Poloidal Loop)
print("\n--- Running Accuracy Validation ---")

# 1. Analyze Plasma Dimensions
R_source = np.sqrt(X**2 + Y**2)
Z_source = Z

# Find geometric center of the plasma cross-section
R_c = (R_source.min() + R_source.max()) / 2.0
Z_c = (Z_source.min() + Z_source.max()) / 2.0

# Find approximate minor radius (max distance from center)
dist_from_center = np.sqrt((R_source - R_c)**2 + (Z_source - Z_c)**2)
a_minor = np.max(dist_from_center)

# 2. Define Scan Geometry: Poloidal Ring outside plasma
# Place it at 1.2x the plasma radius to be safely in the vacuum region
scan_radius = a_minor * 1.2
print(f"    Plasma Minor Radius: {a_minor:.2f} m")
print(f"    Scan Loop Radius:    {scan_radius:.2f} m")

# Verify this is within the Vico-Greengard cutoff (Box L / 2)
R_cutoff = box.Lx / 2.0
if scan_radius >= R_cutoff:
    print("    WARNING: Scan radius exceeds Vico-Greengard cutoff!")
    print("             Increasing box padding is recommended.")
else:
    print(f"    Safety Margin:       {scan_radius:.2f} m < {R_cutoff:.2f} m (Limit)")

# Generate points on the ring (Theta 0 to 2pi)
n_scan = 100
theta_scan = np.linspace(0, 2 * np.pi, n_scan)
scan_R = R_c + scan_radius * np.cos(theta_scan)
scan_Z = Z_c + scan_radius * np.sin(theta_scan)
scan_Y = np.zeros_like(scan_R)  # Toroidal angle phi = 0

# Convert to Box Coordinates (shift by 'center')
targets_local = np.stack([scan_R, scan_Y, scan_Z], axis=1)
targets_box = targets_local + center

# 3. Compute Fields
print("    Computing Direct Biot-Savart (Reference)...")
B_direct = biot_savart_direct(
    jnp.array(targets_box), 
    Xb, Yb, Zb, Jx, Jy, Jz, w
)

print("    Computing NUFFT Spectral Readout...")
bx_n, by_n, bz_n = eval_B_at_targets(
    Bx_hat, By_hat, Bz_hat, jnp.array(targets_box), box
)
B_nufft = np.stack([bx_n, by_n, bz_n], axis=1)

# 4. Error Metrics
diff = B_nufft - B_direct
err = np.linalg.norm(diff, axis=1)
mag = np.linalg.norm(B_direct, axis=1)
rel_err = err / (mag.max() + 1e-12)

print(f"    Max Absolute Error: {np.max(err):.4e} T")
print(f"    Max Relative Error: {np.max(rel_err):.4e}")


from nufft_biot.utils.fast_tracing import trace_field_line_jax

# G. Field Line Tracing (Visualization)
print("\n--- Generating Field Line Trace (JAX Accelerated) ---")

# 1. Prepare Grids for JAX (Keep them on GPU/Device)
# Note: map_coordinates expects (x, y, z) index order. 
# Depending on how ifftn outputs, it is usually (Nx, Ny, Nz).
Bx_grid = jnp.fft.ifftn(Bx_hat).real
By_grid = jnp.fft.ifftn(By_hat).real
Bz_grid = jnp.fft.ifftn(Bz_hat).real

# 2. Define Seeding
seeds = []
step_sizes = []

for r in np.linspace(R_c + 0.4, R_c + a_minor * 0.9, 8):
    seed = center + np.array([r, 0.0, 0.0]) 
    seeds.append(jnp.array(seed)) # Make sure seed is JAX array
    
    # Adaptive step size logic
    radius = abs(r - R_c)
    ds_adaptive = np.clip(radius * 0.2, 0.005, 0.05)
    step_sizes.append(ds_adaptive)

# 3. Run Fast Trace
field_lines = []

# First run triggers compilation (might take ~1 sec), subsequent runs are instant.
print("    JIT Compiling and Tracing...")

for seed, ds in zip(seeds, step_sizes):
    # We can perform huge traces now because it's fast
    line = trace_field_line_jax(
        Bx_grid, By_grid, Bz_grid, 
        seed, 
        box.Lx, box.Ly, box.Lz,
        ds=ds, 
        n_steps=10000,   # Longer trace for better closed loops
        order=3          # Cubic interpolation (Fixes the "thick" drift!)
    )
    field_lines.append(np.array(line)) # Convert back to numpy for plotting

print("    Done.")

# -----------------------------------------------------------------------------
# 3. Plotting
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 6))

# Subplot 1: Field Lines (Poloidal Cross-section)
ax1 = fig.add_subplot(1, 2, 1)
for line in field_lines:
    lc = line - center
    R_line = np.sqrt(lc[:,0]**2 + lc[:,1]**2)
    Z_line = lc[:,2]
    ax1.plot(R_line, Z_line)

# Plot current sources (downsampled for speed)
ax1.scatter(np.sqrt(X**2 + Y**2)[::20], Z[::20], s=0.1, c='k', alpha=0.1, label="Current Source")

# Plot the validation scan ring
ax1.plot(scan_R, scan_Z, 'r--', lw=2, label="Validation Loop")

ax1.axis('equal')
ax1.set_xlabel("R [m]")
ax1.set_ylabel("Z [m]")
ax1.set_title("Field Line Tracing (NUFFT)")
ax1.grid(True)
ax1.legend(loc='upper right')

# Subplot 2: Accuracy Comparison on Loop
ax2 = fig.add_subplot(1, 2, 2)
# Magnitude comparison
ax2.plot(theta_scan, np.linalg.norm(B_direct, axis=1), 'k-', lw=2, label="Direct Biot-Savart")
ax2.plot(theta_scan, np.linalg.norm(B_nufft, axis=1), 'r--', lw=2, label="NUFFT (Spectral)")

# Error on secondary axis
ax2_right = ax2.twinx()
ax2_right.semilogy(theta_scan, rel_err, 'b:', label="Relative Error")
ax2_right.set_ylabel("Relative Error", color='b')
ax2_right.tick_params(axis='y', labelcolor='b')

ax2.set_xlabel("Poloidal Angle (radians)")
ax2.set_ylabel("|B| [Tesla]")
ax2.set_title(f"Validation on Loop (r = {scan_radius:.2f} m)")
ax2.set_xlim(0, 2*np.pi)
ax2.grid(True)

# Merge legends
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_right.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

plt.tight_layout()
plt.show()