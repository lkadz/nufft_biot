import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from nufft_biot.forward import forward_B
from nufft_biot.types import BoxParams
from nufft_biot.utils import PeriodicFieldInterpolator, trace_field_line_rk4

jax.config.update("jax_enable_x64", True)


major_radius = 1.0
minor_radius = 0.3
I = 1e5

box = BoxParams(
    Lx=10.0,
    Ly=10.0,
    Lz=10.0,
    Nx=128,
    Ny=128,
    Nz=128,
)

x = jnp.array([major_radius])

Bx, By, Bz, center = forward_B(
    x,
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

for r in np.linspace(0.2, 0.9 * minor_radius, 4):
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
    R = np.sqrt(flc[:, 0] ** 2 + flc[:, 1] ** 2)
    Z = flc[:, 2]
    ax.plot(R, Z, linewidth=1.5)

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
