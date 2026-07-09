import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _opencv_lens_distortion_delta(
    u: Float[Array, ""], v: Float[Array, ""], k1: float, k2: float, p1: float, p2: float
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Distortion *offset* (not the distorted point itself): distorted =
    (u,v) + delta(u,v). Matches instant-ngp's opencv_lens_distortion_delta
    (common_device.cuh:267) exactly."""
    u2, v2 = u * u, v * v
    uv = u * v
    r2 = u2 + v2
    radial = k1 * r2 + k2 * r2 * r2
    du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2)
    dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2)
    return du, dv


def undistort_normalized(
    xd: Float[Array, ""],
    yd: Float[Array, ""],
    k1: float,
    k2: float,
    p1: float,
    p2: float,
    max_iters: int = 100,
    max_step_norm2: float = 1e-10,
    rel_step_size: float = 1e-6,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Invert Brown-Conrady radial-tangential distortion via Newton's
    method with a numerically-differentiated (5-point central-difference)
    Jacobian and early-exit-on-convergence, matching instant-ngp's
    `iterative_lens_undistortion` (common_device.cuh:307) exactly —
    including using finite differences rather than an analytic Jacobian
    (their template is shared with fisheye distortion, so a generic
    numerical Jacobian avoids hand-deriving one per distortion model) and
    a genuine early-exit loop (up to 100 iterations, breaks once the step
    norm is small — this runs GPU-side in the original, so unlike the
    NeRF marcher there's no SIMT-lockstep reason to avoid a data-dependent
    loop here)."""
    eps = jnp.finfo(jnp.float32).eps
    x0 = jnp.array([xd, yd])

    def cond_fn(carry):
        _, step_norm2, i = carry
        return (i < max_iters) & (step_norm2 >= max_step_norm2)

    def body_fn(carry):
        x, _, i = carry
        u, v = x[0], x[1]
        step0 = jnp.maximum(eps, jnp.abs(rel_step_size * u))
        step1 = jnp.maximum(eps, jnp.abs(rel_step_size * v))

        du, dv = _opencv_lens_distortion_delta(u, v, k1, k2, p1, p2)
        du_0b, dv_0b = _opencv_lens_distortion_delta(u - step0, v, k1, k2, p1, p2)
        du_0f, dv_0f = _opencv_lens_distortion_delta(u + step0, v, k1, k2, p1, p2)
        du_1b, dv_1b = _opencv_lens_distortion_delta(u, v - step1, k1, k2, p1, p2)
        du_1f, dv_1f = _opencv_lens_distortion_delta(u, v + step1, k1, k2, p1, p2)

        m00 = 1.0 + (du_0f - du_0b) / (2 * step0)
        m01 = (du_1f - du_1b) / (2 * step1)
        m10 = (dv_0f - dv_0b) / (2 * step0)
        m11 = 1.0 + (dv_1f - dv_1b) / (2 * step1)

        rx = u + du - xd
        ry = v + dv - yd

        det = m00 * m11 - m01 * m10
        det = jnp.where(jnp.abs(det) < 1e-20, 1e-20, det)
        step_x0 = (rx * m11 - ry * m01) / det
        step_x1 = (m00 * ry - m10 * rx) / det

        new_x = jnp.array([u - step_x0, v - step_x1])
        step_norm2 = step_x0 * step_x0 + step_x1 * step_x1
        return (new_x, step_norm2, i + 1)

    init = (x0, jnp.asarray(jnp.inf), jnp.asarray(0, jnp.int32))
    x, _, _ = jax.lax.while_loop(cond_fn, body_fn, init)
    return x[0], x[1]


def camera_ray(
    c2w: Float[Array, "4 4"],
    px: Float[Array, ""],
    py: Float[Array, ""],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float = 0.0,
    k2: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> tuple[Float[Array, "3"], Float[Array, "3"]]:
    """Pinhole + Brown-Conrady camera ray in world space. OpenGL/NeRF convention:
    camera looks down -z, +y is up, c2w is camera-to-world."""
    xd = (px - cx) / fx
    yd = (py - cy) / fy
    xu, yu = undistort_normalized(xd, yd, k1, k2, p1, p2)

    dir_cam = jnp.array([xu, -yu, -1.0])
    d_world = c2w[:3, :3] @ dir_cam
    d_world = d_world / jnp.linalg.norm(d_world)
    o_world = c2w[:3, 3]
    return o_world, d_world


def look_at_c2w(
    eye: Float[Array, "3"], target: Float[Array, "3"], up: Float[Array, "3"]
) -> Float[Array, "4 4"]:
    """Camera-to-world matrix for a camera at `eye` looking at `target`,
    OpenGL/NeRF convention (camera looks down local -z, +y is up)."""
    forward = target - eye
    forward = forward / jnp.linalg.norm(forward)
    right = jnp.cross(forward, up)
    right = right / jnp.linalg.norm(right)
    true_up = jnp.cross(right, forward)
    c2w = jnp.eye(4)
    c2w = c2w.at[:3, 0].set(right)
    c2w = c2w.at[:3, 1].set(true_up)
    c2w = c2w.at[:3, 2].set(-forward)
    c2w = c2w.at[:3, 3].set(eye)
    return c2w


def orbit_camera_rays(
    c2w: Float[Array, "4 4"], fx: float, fy: float, cx: float, cy: float, h: int, w: int
) -> tuple[Float[Array, "h*w 3"], Float[Array, "h*w 3"]]:
    """Full-frame pinhole rays for a single camera pose (no distortion) —
    used by the non-dataset-driven Volume/SDF preview renderers."""
    ys, xs = jnp.meshgrid(
        jnp.arange(h, dtype=jnp.float32), jnp.arange(w, dtype=jnp.float32), indexing="ij"
    )
    px, py = xs.reshape(-1), ys.reshape(-1)
    dir_cam = jnp.stack([(px - cx) / fx, -(py - cy) / fy, -jnp.ones_like(px)], axis=-1)
    d_world = dir_cam @ c2w[:3, :3].T
    d_world = d_world / jnp.linalg.norm(d_world, axis=-1, keepdims=True)
    o_world = jnp.broadcast_to(c2w[:3, 3], d_world.shape)
    return o_world, d_world


def orbit_c2w(center: Float[Array, "3"], radius: float, azimuth_deg: float, elevation_deg: float) -> Float[Array, "4 4"]:
    az, el = jnp.radians(azimuth_deg), jnp.radians(elevation_deg)
    eye = center + radius * jnp.array(
        [jnp.cos(el) * jnp.cos(az), jnp.sin(el), jnp.cos(el) * jnp.sin(az)]
    )
    return look_at_c2w(eye, center, jnp.array([0.0, 1.0, 0.0]))
