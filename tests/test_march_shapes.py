"""Lightweight shape/dtype/NaN sanity checks for the ray marchers (NeRF
adaptive marcher, SDF sphere tracer, Volume marcher). Not a pytest suite —
a fast pre-flight script, run directly:
    JAX_PLATFORMS=cpu python tests/test_march_shapes.py
"""

import chex
import jax
import jax.numpy as jnp

from jaxingp.config import NerfNetworkConfig, SdfNetworkConfig
from jaxingp.data.voxel_dataset import synthesize_toy_volume
from jaxingp.geometry.aabb import BoundingBox
from jaxingp.nn.nerf_network import NerfNetwork
from jaxingp.nn.sdf_network import SdfNetwork
from jaxingp.occupancy.grid import OccupancyGrid
from jaxingp.render.march import march_rays
from jaxingp.render.render import render_rays_adaptive
from jaxingp.render.sphere_trace import render_rays_sdf, sphere_trace_ray
from jaxingp.render.volume_march import render_rays_volume


def random_rays(key, n_rays, center=0.5, radius=0.3):
    o_key, d_key = jax.random.split(key)
    rays_o = center + jax.random.uniform(o_key, (n_rays, 3), minval=-radius, maxval=radius)
    rays_d = jax.random.normal(d_key, (n_rays, 3))
    rays_d = rays_d / jnp.linalg.norm(rays_d, axis=-1, keepdims=True)
    return rays_o, rays_d


def test_nerf_adaptive_marcher():
    key = jax.random.PRNGKey(0)
    model = NerfNetwork(key, NerfNetworkConfig())
    grid = OccupancyGrid(grid_size=16, n_cascades=4)
    # seed fully occupied so this is a pure shape/NaN check, independent of
    # occupancy-grid semantics (covered separately by the training runs)
    grid = chex_replace_density(grid, jnp.ones_like(grid.density) * 1e4)
    aabb = BoundingBox()

    n_rays, max_samples, max_cascade = 32, 16, grid.n_cascades - 1
    rays_o, rays_d = random_rays(key, n_rays)
    march = march_rays(rays_o, rays_d, grid, aabb, max_samples, 256, 1.0 / 256, max_cascade, 1e-3)
    chex.assert_shape(march.positions, (n_rays, max_samples, 3))
    chex.assert_shape(march.n_valid, (n_rays,))
    chex.assert_tree_all_finite(march.positions)
    assert bool(jnp.all(march.n_valid >= 0)) and bool(jnp.all(march.n_valid <= max_samples))

    rgb, acc, n_valid = render_rays_adaptive(
        model, grid, aabb, rays_o, rays_d, max_samples, 256, 1.0 / 256, max_cascade, 1e-3, jnp.ones(3)
    )
    chex.assert_shape(rgb, (n_rays, 3))
    chex.assert_tree_all_finite(rgb)
    chex.assert_tree_all_finite(acc)
    print("nerf adaptive marcher: shapes/finite ok")


def chex_replace_density(grid, density):
    import equinox as eqx

    return eqx.tree_at(lambda g: g.density, grid, density)


def test_sdf_sphere_tracer():
    key = jax.random.PRNGKey(1)
    model = SdfNetwork(key, SdfNetworkConfig())
    aabb = BoundingBox(min_corner=-jnp.ones(3), max_corner=jnp.ones(3))

    n_rays = 32
    rays_o = jnp.zeros((n_rays, 3)) + jnp.array([0.0, 0.0, 3.0])
    dirs = jax.random.normal(key, (n_rays, 3)) * 0.1 + jnp.array([0.0, 0.0, -1.0])
    rays_d = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)

    result = jax.vmap(lambda o, d: sphere_trace_ray(o, d, model, aabb, max_iters=64))(rays_o, rays_d)
    chex.assert_shape(result.hit_pos, (n_rays, 3))
    chex.assert_shape(result.hit, (n_rays,))
    chex.assert_tree_all_finite(result.hit_pos)

    rgb = render_rays_sdf(model, aabb, rays_o, rays_d, jnp.array([0.5, 0.7, 0.5]), jnp.ones(3), max_iters=64)
    chex.assert_shape(rgb, (n_rays, 3))
    chex.assert_tree_all_finite(rgb)
    print("sdf sphere tracer: shapes/finite ok")


def test_volume_marcher():
    volume = synthesize_toy_volume(resolution=16)
    aabb = BoundingBox()
    n_rays = 32
    key = jax.random.PRNGKey(2)
    rays_o, rays_d = random_rays(key, n_rays, center=0.5, radius=0.6)

    rgb = render_rays_volume(volume, aabb, rays_o, rays_d, n_samples=32, background=jnp.ones(3))
    chex.assert_shape(rgb, (n_rays, 3))
    chex.assert_tree_all_finite(rgb)
    print("volume marcher: shapes/finite ok")


if __name__ == "__main__":
    test_nerf_adaptive_marcher()
    test_sdf_sphere_tracer()
    test_volume_marcher()
    print("all march checks passed")
