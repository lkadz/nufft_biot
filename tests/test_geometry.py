import jax.numpy as jnp

from nufft_biot.geometry import geom_from_x


def test_geom_shapes():
    x = jnp.array([0.0])
    N_rho, N_theta, N_zeta = 4, 8, 16
    R0 = 1.2
    a = 0.3

    X, Y, Z, rho, theta, zeta, w = geom_from_x(
        x,
        N_rho=N_rho,
        N_theta=N_theta,
        N_zeta=N_zeta,
        major_radius=R0,
        minor_radius=a,
    )

    N = N_rho * N_theta * N_zeta

    assert X.shape == (N,)
    assert Y.shape == (N,)
    assert Z.shape == (N,)
    assert rho.shape == (N,)
    assert theta.shape == (N,)
    assert zeta.shape == (N,)
    assert w.shape == (N,)


def test_geom_inside_torus():
    x = jnp.array([0.0])
    N_rho, N_theta, N_zeta = 6, 12, 24
    R0 = 1.0
    a = 0.25

    X, Y, Z, *_ = geom_from_x(
        x,
        N_rho=N_rho,
        N_theta=N_theta,
        N_zeta=N_zeta,
        major_radius=R0,
        minor_radius=a,
    )

    R = jnp.sqrt(X**2 + Y**2)
    dist2 = (R - R0) ** 2 + Z**2

    assert jnp.all(dist2 <= a**2 + 1e-14)


def test_geom_volume():
    x = jnp.array([0.0])
    N_rho, N_theta, N_zeta = 8, 16, 32
    R0 = 1.3
    a = 0.4

    _, _, _, _, _, _, w = geom_from_x(
        x,
        N_rho=N_rho,
        N_theta=N_theta,
        N_zeta=N_zeta,
        major_radius=R0,
        minor_radius=a,
    )

    V_num = jnp.sum(w)
    V_exact = 2.0 * jnp.pi**2 * R0 * a**2

    assert jnp.allclose(V_num, V_exact, rtol=5e-3)


def test_geom_center_of_mass():
    x = jnp.array([0.0])
    N_rho, N_theta, N_zeta = 6, 12, 24
    R0 = 1.1
    a = 0.35

    X, Y, Z, _, _, _, w = geom_from_x(
        x,
        N_rho=N_rho,
        N_theta=N_theta,
        N_zeta=N_zeta,
        major_radius=R0,
        minor_radius=a,
    )

    assert jnp.allclose(jnp.sum(X * w), 0.0, atol=1e-12)
    assert jnp.allclose(jnp.sum(Y * w), 0.0, atol=1e-12)
    assert jnp.allclose(jnp.sum(Z * w), 0.0, atol=1e-12)


def test_geom_cylindrical_consistency():
    x = jnp.array([0.0])
    N_rho, N_theta, N_zeta = 5, 10, 20
    R0 = 0.9
    a = 0.2

    X, Y, _, _, _, _, _ = geom_from_x(
        x,
        N_rho=N_rho,
        N_theta=N_theta,
        N_zeta=N_zeta,
        major_radius=R0,
        minor_radius=a,
    )

    R = jnp.sqrt(X**2 + Y**2)
    assert jnp.all(R > 0.0)


def test_geom_jacobian_pointwise():
    x = jnp.array([0.0])
    N_rho, N_theta, N_zeta = 6, 12, 24
    R0 = 1.0
    a = 0.3

    X, Y, Z, rho, theta, zeta, w = geom_from_x(
        x,
        N_rho=N_rho,
        N_theta=N_theta,
        N_zeta=N_zeta,
        major_radius=R0,
        minor_radius=a,
    )

    drho = 1.0 / N_rho
    dtheta = 2.0 * jnp.pi / N_theta
    dzeta = 2.0 * jnp.pi / N_zeta

    jac_numeric = w / (drho * dtheta * dzeta)
    jac_exact = a**2 * rho * (R0 + a * rho * jnp.cos(theta))

    assert jnp.allclose(jac_numeric, jac_exact, rtol=1e-12)


def test_geom_volume_convergence():
    x = jnp.array([0.0])
    R0 = 1.2
    a = 0.25

    _, _, _, _, _, _, w_coarse = geom_from_x(
        x,
        N_rho=4,
        N_theta=8,
        N_zeta=16,
        major_radius=R0,
        minor_radius=a,
    )

    _, _, _, _, _, _, w_fine = geom_from_x(
        x,
        N_rho=8,
        N_theta=16,
        N_zeta=32,
        major_radius=R0,
        minor_radius=a,
    )

    V_exact = 2.0 * jnp.pi**2 * R0 * a**2

    err_coarse = jnp.abs(jnp.sum(w_coarse) - V_exact)
    err_fine = jnp.abs(jnp.sum(w_fine) - V_exact)

    assert err_fine < err_coarse


def test_geom_R_theta_identity():
    x = jnp.array([0.0])
    N_rho, N_theta, N_zeta = 5, 10, 20
    R0 = 1.1
    a = 0.4

    X, Y, _, rho, theta, _, _ = geom_from_x(
        x,
        N_rho=N_rho,
        N_theta=N_theta,
        N_zeta=N_zeta,
        major_radius=R0,
        minor_radius=a,
    )

    R = jnp.sqrt(X**2 + Y**2)
    R_expected = R0 + a * rho * jnp.cos(theta)

    assert jnp.allclose(R, R_expected, atol=1e-12)


def test_geom_theta_sine_symmetry():
    x = jnp.array([0.0])
    N_rho, N_theta, N_zeta = 6, 12, 24
    R0 = 1.0
    a = 0.3

    _, _, _, _, theta, _, w = geom_from_x(
        x,
        N_rho=N_rho,
        N_theta=N_theta,
        N_zeta=N_zeta,
        major_radius=R0,
        minor_radius=a,
    )

    assert jnp.allclose(jnp.sum(jnp.sin(theta) * w), 0.0, atol=1e-12)
