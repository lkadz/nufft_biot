import jax.numpy as jnp
from nufft_biot.current import compute_J_components_from_x_and_geom


def test_current_support_inside_outside():
    R0 = 1.0
    a = 0.3
    I = 2.5

    zeta = jnp.array([0.0, 0.0])
    R = jnp.array([R0, R0 + 2 * a])
    Z = jnp.array([0.0, 0.0])

    Jx, Jy, Jz = compute_J_components_from_x_and_geom(
        x=jnp.array([0.0]),
        R=R,
        Z=Z,
        rho_nodes=jnp.zeros_like(R),
        theta_nodes=jnp.zeros_like(R),
        zeta=zeta,
        I=I,
        major_radius=R0,
        minor_radius=a,
    )

    assert jnp.all(Jz == 0.0)
    assert Jx[0]**2 + Jy[0]**2 > 0.0
    assert Jx[1] == 0.0
    assert Jy[1] == 0.0


def test_current_purely_toroidal():
    R0 = 1.2
    a = 0.4
    I = 1.0

    zeta = jnp.linspace(0.0, 2 * jnp.pi, 50, endpoint=False)
    R = jnp.full_like(zeta, R0)
    Z = jnp.zeros_like(zeta)

    Jx, Jy, Jz = compute_J_components_from_x_and_geom(
        x=jnp.array([0.0]),
        R=R,
        Z=Z,
        rho_nodes=jnp.zeros_like(R),
        theta_nodes=jnp.zeros_like(R),
        zeta=zeta,
        I=I,
        major_radius=R0,
        minor_radius=a,
    )

    radial_dot = Jx * jnp.cos(zeta) + Jy * jnp.sin(zeta)

    assert jnp.allclose(radial_dot, 0.0, atol=1e-12)
    assert jnp.all(Jz == 0.0)


def test_current_uniform_magnitude():
    R0 = 0.9
    a = 0.25
    I = 3.0

    zeta = jnp.linspace(0.0, 2 * jnp.pi, 100, endpoint=False)
    R = jnp.full_like(zeta, R0)
    Z = jnp.zeros_like(zeta)

    Jx, Jy, _ = compute_J_components_from_x_and_geom(
        x=jnp.array([0.0]),
        R=R,
        Z=Z,
        rho_nodes=jnp.zeros_like(R),
        theta_nodes=jnp.zeros_like(R),
        zeta=zeta,
        I=I,
        major_radius=R0,
        minor_radius=a,
    )

    Jmag = jnp.sqrt(Jx**2 + Jy**2)
    J_expected = I / (jnp.pi * a**2)

    assert jnp.allclose(Jmag, J_expected, rtol=1e-12)


def test_current_zero_outside_torus():
    R0 = 1.0
    a = 0.2
    I = 1.0

    zeta = jnp.linspace(0.0, 2 * jnp.pi, 20, endpoint=False)
    R = jnp.full_like(zeta, R0 + 3 * a)
    Z = jnp.zeros_like(zeta)

    Jx, Jy, Jz = compute_J_components_from_x_and_geom(
        x=jnp.array([0.0]),
        R=R,
        Z=Z,
        rho_nodes=jnp.zeros_like(R),
        theta_nodes=jnp.zeros_like(R),
        zeta=zeta,
        I=I,
        major_radius=R0,
        minor_radius=a,
    )

    assert jnp.all(Jx == 0.0)
    assert jnp.all(Jy == 0.0)
    assert jnp.all(Jz == 0.0)


def test_total_current_cross_section():
    x = jnp.array([0.0])
    R0 = 1.1
    a = 0.35
    I = 2.0

    N_rho, N_theta = 24, 48

    rho = (jnp.arange(N_rho) + 0.5) / N_rho
    theta = (jnp.arange(N_theta) + 0.5) * (2 * jnp.pi / N_theta)

    rho, theta = jnp.meshgrid(rho, theta, indexing="ij")

    R = R0 + a * rho * jnp.cos(theta)
    Z = a * rho * jnp.sin(theta)
    zeta = jnp.zeros_like(R)

    Jx, Jy, _ = compute_J_components_from_x_and_geom(
        x=x,
        R=R.reshape(-1),
        Z=Z.reshape(-1),
        rho_nodes=rho.reshape(-1),
        theta_nodes=theta.reshape(-1),
        zeta=zeta.reshape(-1),
        I=I,
        major_radius=R0,
        minor_radius=a,
    )

    Jphi = -Jx * jnp.sin(zeta.reshape(-1)) + Jy * jnp.cos(zeta.reshape(-1))

    drho = 1.0 / N_rho
    dtheta = 2 * jnp.pi / N_theta
    dA = a**2 * rho.reshape(-1) * drho * dtheta

    I_num = jnp.sum(Jphi * dA)

    assert jnp.allclose(I_num, I, rtol=5e-3)


def test_current_independent_of_unused_inputs():
    R0 = 1.0
    a = 0.3
    I = 1.0

    zeta = jnp.linspace(0.0, 2 * jnp.pi, 30, endpoint=False)
    R = jnp.full_like(zeta, R0)
    Z = jnp.zeros_like(zeta)

    rho1 = jnp.zeros_like(zeta)
    rho2 = jnp.ones_like(zeta)

    Jx1, Jy1, Jz1 = compute_J_components_from_x_and_geom(
        x=jnp.array([1.0]),
        R=R,
        Z=Z,
        rho_nodes=rho1,
        theta_nodes=rho1,
        zeta=zeta,
        I=I,
        major_radius=R0,
        minor_radius=a,
    )

    Jx2, Jy2, Jz2 = compute_J_components_from_x_and_geom(
        x=jnp.array([-3.0]),
        R=R,
        Z=Z,
        rho_nodes=rho2,
        theta_nodes=rho2,
        zeta=zeta,
        I=I,
        major_radius=R0,
        minor_radius=a,
    )

    assert jnp.allclose(Jx1, Jx2)
    assert jnp.allclose(Jy1, Jy2)
    assert jnp.allclose(Jz1, Jz2)
