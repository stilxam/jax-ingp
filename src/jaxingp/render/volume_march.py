import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxingp.geometry.aabb import BoundingBox
from jaxingp.render.composite import composite_ray


def trilinear_sample_voxel(
    volume: Float[Array, "D H W 4"], pos: Float[Array, "3"]
) -> Float[Array, "4"]:
    """Trilinear lookup into a dense (D, H, W, 4) RGB+density voxel grid at
    a continuous position in [0,1]^3. Same corner-weighted gather idea as
    the hash-grid encoding, but direct dense indexing (no hashing) since
    the whole point of this primitive is that the grid is precomputed and
    small enough to index densely."""
    d, h, w, _ = volume.shape
    res = jnp.array([d - 1, h - 1, w - 1], dtype=jnp.float32)
    coords = jnp.clip(pos, 0.0, 1.0) * res
    i0 = jnp.floor(coords).astype(jnp.int32)
    frac = coords - i0

    def corner(dz, dy, dx):
        idx = jnp.clip(i0 + jnp.array([dz, dy, dx]), 0, jnp.array([d - 1, h - 1, w - 1]))
        weight = (
            (frac[0] if dz else 1 - frac[0])
            * (frac[1] if dy else 1 - frac[1])
            * (frac[2] if dx else 1 - frac[2])
        )
        return weight * volume[idx[0], idx[1], idx[2]]

    out = jnp.zeros((4,))
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                out = out + corner(dz, dy, dx)
    return out


def render_ray_volume(
    volume: Float[Array, "D H W 4"],
    aabb: BoundingBox,
    ray_o: Float[Array, "3"],
    ray_d: Float[Array, "3"],
    n_samples: int,
    background: Float[Array, "3"],
) -> Float[Array, "3"]:
    """Fixed-count stratified marcher (no occupancy grid — the whole grid is
    already small and precomputed, so there's no empty space worth skipping)
    with a trilinear voxel lookup at each sample instead of an MLP forward
    pass, then the same emission-absorption compositing as NeRF."""
    t0, t1 = aabb.ray_intersect(ray_o, ray_d)
    t0 = jnp.maximum(t0, 1e-4)
    t1 = jnp.maximum(t1, t0 + 1e-3)

    t = t0 + (t1 - t0) * (jnp.arange(n_samples, dtype=jnp.float32) + 0.5) / n_samples
    positions = ray_o[None, :] + t[:, None] * ray_d[None, :]

    rgbd = jax.vmap(trilinear_sample_voxel, in_axes=(None, 0))(volume, positions)
    rgb = rgbd[:, :3]
    density = jax.nn.relu(rgbd[:, 3])

    dt = (t1 - t0) / n_samples
    dts = jnp.full((n_samples,), dt)
    valid = jnp.ones((n_samples,), dtype=bool)

    rgb_out, _ = composite_ray(rgb, density, dts, valid, background)
    return rgb_out


def render_rays_volume(
    volume: Float[Array, "D H W 4"],
    aabb: BoundingBox,
    rays_o: Float[Array, "n_rays 3"],
    rays_d: Float[Array, "n_rays 3"],
    n_samples: int,
    background: Float[Array, "3"],
) -> Float[Array, "n_rays 3"]:
    return jax.vmap(render_ray_volume, in_axes=(None, None, 0, 0, None, None))(
        volume, aabb, rays_o, rays_d, n_samples, background
    )
