import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from nufft_biot.forward import forward_B
from nufft_biot.types import BoxParams
from nufft_biot.utils import PeriodicFieldInterpolator, trace_field_line_rk4

jax.config.update("jax_enable_x64", True)


major_radius = 1.0
I = 1e5

box = BoxParams(
    Lx=16.0,
    Ly=16.0,
    Lz=16.0,
    Nx=256,
    Ny=256,
    Nz=256,
)

x = jnp.array([major_radius])

Bx, By, Bz, center = forward_B(
    x,
    box,
    I=I,
    major_radius=major_radius,
    minor_radius=0.0,
    current_model="filament",
    N_zeta=512,
)

Bx = np.asarray(Bx)
By = np.asarray(By)
Bz = np.asarray(Bz)
center = np.asarray(center)

interp = PeriodicFieldInterpolator(Bx, By, Bz, box)

seeds = []
for r in [0.05, 0.1, 0.15]:
    seed = np.array(
        [
            major_radius + r,
            0.0,
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

theta = np.linspace(0.0, 2.0 * np.pi, 400)
coil_x = major_radius * np.cos(theta)
coil_y = major_radius * np.sin(theta)
coil_z = np.zeros_like(theta)

ax.plot(coil_x, coil_y, coil_z, color="magenta", linewidth=3)

ax.set_box_aspect([1, 1, 1])
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Magnetic Field Lines of a Circular Current Filament")

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
ax = plt.gca()

for fl in field_lines:
    flc = fl - center
    R = np.sqrt(flc[:, 0] ** 2 + flc[:, 1] ** 2)
    Z = flc[:, 2]
    ax.plot(R, Z)

ax.plot(
    major_radius * np.ones_like(theta),
    np.zeros_like(theta),
    color="magenta",
    linewidth=3,
)

ax.set_aspect("equal")
ax.set_xlim(0.6, 1.4)
ax.set_ylim(-0.8, 0.8)

ax.set_xlabel("R")
ax.set_ylabel("Z")
ax.set_title("Poloidal Projection of Filament Magnetic Field")

plt.grid(True)
plt.tight_layout()
plt.show()
