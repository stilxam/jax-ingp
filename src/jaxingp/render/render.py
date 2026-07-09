import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from jaxingp.geometry.aabb import BoundingBox
from jaxingp.nn.nerf_network import NerfNetwork
from jaxingp.occupancy.grid import OccupancyGrid
from jaxingp.render.composite import composite_ray
from jaxingp.render.march import march_rays


def render_ray_uniform(
    model: NerfNetwork,
    aabb: BoundingBox,
    ray_o: Float[Array, "3"],
    ray_d: Float[Array, "3"],
    n_samples: int,
    key: PRNGKeyArray,
    background: Float[Array, "3"],
) -> tuple[Float[Array, "3"], Float[Array, ""]]:
    """Temporary Phase-3 sampler: fixed stratified samples between the ray's
    AABB entry/exit, bypassing the occupancy grid / adaptive marcher
    (added in Phase 5). Validates the network + compositing independently."""
    t0, t1 = aabb.ray_intersect(ray_o, ray_d)
    t0 = jnp.maximum(t0, 1e-4)
    t1 = jnp.maximum(t1, t0 + 1e-3)

    u = jnp.arange(n_samples, dtype=jnp.float32) / n_samples
    jitter = jax.random.uniform(key, (n_samples,)) / n_samples
    t = t0 + (t1 - t0) * (u + jitter)

    positions = ray_o[None, :] + t[:, None] * ray_d[None, :]
    dt = (t1 - t0) / n_samples
    dts = jnp.full((n_samples,), dt)
    valid = jnp.ones((n_samples,), dtype=bool)

    rgb, density = jax.vmap(model, in_axes=(0, None))(positions, ray_d)
    return composite_ray(rgb, density, dts, valid, background)


def render_rays_uniform(
    model: NerfNetwork,
    aabb: BoundingBox,
    rays_o: Float[Array, "n_rays 3"],
    rays_d: Float[Array, "n_rays 3"],
    n_samples: int,
    key: PRNGKeyArray,
    background: Float[Array, "3"],
) -> tuple[Float[Array, "n_rays 3"], Float[Array, "n_rays"]]:
    keys = jax.random.split(key, rays_o.shape[0])
    return jax.vmap(render_ray_uniform, in_axes=(None, None, 0, 0, None, 0, None))(
        model, aabb, rays_o, rays_d, n_samples, keys, background
    )


def render_rays_uniform_chunked(
    model: NerfNetwork,
    aabb: BoundingBox,
    rays_o: Float[Array, "n_rays 3"],
    rays_d: Float[Array, "n_rays 3"],
    n_samples: int,
    key: PRNGKeyArray,
    background: Float[Array, "3"],
    chunk_size: int = 8192,
) -> Float[Array, "n_rays 3"]:
    """Renders eval-time full-frame ray batches in chunks to bound memory."""
    n_rays = rays_o.shape[0]
    out = []
    for start in range(0, n_rays, chunk_size):
        end = min(start + chunk_size, n_rays)
        key, sub = jax.random.split(key)
        rgb, _ = render_rays_uniform(
            model, aabb, rays_o[start:end], rays_d[start:end], n_samples, sub, background
        )
        out.append(rgb)
    return jnp.concatenate(out, axis=0)


def render_rays_adaptive(
    model: NerfNetwork,
    grid: OccupancyGrid,
    aabb: BoundingBox,
    rays_o: Float[Array, "n_rays 3"],
    rays_d: Float[Array, "n_rays 3"],
    max_samples: int,
    max_march_iters: int,
    cone_angle_min_stepsize: float,
    near_distance: float,
    background: Float[Array, "3"],
) -> tuple[Float[Array, "n_rays 3"], Float[Array, "n_rays"], Int[Array, "n_rays"]]:
    """Occupancy-grid adaptive marching + compositing. Flattens all rays'
    padded samples into one (n_rays*max_samples,) batch for a single big
    NerfNetwork forward pass, then reshapes back per-ray for compositing —
    the padded/masked layout makes this a trivial reshape, unlike CUDA's
    ray-indexed access pattern which needs explicit numsteps/base offsets."""
    march = march_rays(
        rays_o, rays_d, grid, aabb, max_samples, max_march_iters, cone_angle_min_stepsize, near_distance,
    )
    n_rays = rays_o.shape[0]
    flat_pos = march.positions.reshape(-1, 3)
    flat_dir = march.dirs.reshape(-1, 3)
    rgb_flat, density_flat = jax.vmap(model)(flat_pos, flat_dir)
    rgb = rgb_flat.reshape(n_rays, max_samples, 3)
    density = density_flat.reshape(n_rays, max_samples)

    rgb_out, acc = jax.vmap(composite_ray, in_axes=(0, 0, 0, 0, None))(
        rgb, density, march.dts, march.valid, background
    )
    return rgb_out, acc, march.n_valid


def render_rays_adaptive_chunked(
    model: NerfNetwork,
    grid: OccupancyGrid,
    aabb: BoundingBox,
    rays_o: Float[Array, "n_rays 3"],
    rays_d: Float[Array, "n_rays 3"],
    max_samples: int,
    max_march_iters: int,
    cone_angle_min_stepsize: float,
    near_distance: float,
    background: Float[Array, "3"],
    chunk_size: int = 8192,
) -> tuple[Float[Array, "n_rays 3"], Int[Array, "n_rays"]]:
    n_rays = rays_o.shape[0]
    rgb_out, n_valid_out = [], []
    for start in range(0, n_rays, chunk_size):
        end = min(start + chunk_size, n_rays)
        rgb, _, n_valid = render_rays_adaptive(
            model, grid, aabb, rays_o[start:end], rays_d[start:end],
            max_samples, max_march_iters, cone_angle_min_stepsize, near_distance, background,
        )
        rgb_out.append(rgb)
        n_valid_out.append(n_valid)
    return jnp.concatenate(rgb_out, axis=0), jnp.concatenate(n_valid_out, axis=0)
