import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from jaxingp.geometry.aabb import BoundingBox
from jaxingp.occupancy.grid import OccupancyGrid, is_occupied, mip_from_pos


class MarchResult(eqx.Module):
    positions: Float[Array, "max_samples 3"]
    dirs: Float[Array, "max_samples 3"]
    dts: Float[Array, "max_samples"]
    valid: Bool[Array, "max_samples"]
    n_valid: Int[Array, ""]


class _Carry(eqx.Module):
    t: Float[Array, ""]
    n_valid: Int[Array, ""]
    positions: Float[Array, "max_samples 3"]
    dirs: Float[Array, "max_samples 3"]
    dts: Float[Array, "max_samples"]
    valid: Bool[Array, "max_samples"]
    n_iter: Int[Array, ""]


def _distance_to_next_voxel(
    pos: Float[Array, "3"], ray_d: Float[Array, "3"], mip: Int[Array, ""], grid_size: int
) -> Float[Array, ""]:
    """DDA distance (in ray-parameter units) to the next voxel boundary of
    the cascaded grid at cascade `mip`, ported faithfully from the CUDA
    `distance_to_next_voxel`/`advance_to_next_voxel` geometry (empty-space
    skipping efficiency depends on exact voxel-boundary jumps, unlike the
    step-size formula, which we simplified — see render/march.py docstring)."""
    scale = 2.0 ** (-mip.astype(jnp.float32))
    p_scaled = ((pos - 0.5) * scale + 0.5) * grid_size
    d_scaled = ray_d * scale * grid_size

    t_pos = jnp.where(d_scaled > 0, (jnp.floor(p_scaled) + 1 - p_scaled) / d_scaled, jnp.inf)
    t_neg = jnp.where(d_scaled < 0, (jnp.ceil(p_scaled) - 1 - p_scaled) / d_scaled, jnp.inf)
    return jnp.min(jnp.minimum(t_pos, t_neg)) + 1e-6


def march_ray(
    ray_o: Float[Array, "3"],
    ray_d: Float[Array, "3"],
    grid: OccupancyGrid,
    aabb: BoundingBox,
    max_samples: int,
    max_march_iters: int,
    cone_angle_min_stepsize: float,
    near_distance: float,
) -> MarchResult:
    """Per-ray adaptive marcher: `jax.lax.while_loop`, vmapped externally
    across rays (see march_rays). Each ray owns a fixed-size padded output
    slot (positions/dirs/dts/valid of length max_samples) rather than a
    globally-compacted buffer — see PLAN.md "Ray marching" section for why
    this is the JAX-native substitute for CUDA's atomicAdd-compaction scheme.

    Step-size simplification (explicit deviation from CUDA): dt grows
    geometrically with cascade mip (dt = cone_angle_min_stepsize * 2**mip)
    instead of the closed-form log-space `to_stepping_space`/`advance_n_steps`
    formulas, which exist only to support jumping an arbitrary number of
    steps in O(1) for the compaction scheme we don't use. Voxel-boundary
    skipping (`_distance_to_next_voxel`) *is* ported exactly since that's
    what makes empty-space skipping efficient.
    """
    t0, t1 = aabb.ray_intersect(ray_o, ray_d)
    t0 = jnp.maximum(t0, near_distance)

    init = _Carry(
        t=t0,
        n_valid=jnp.zeros((), jnp.int32),
        positions=jnp.zeros((max_samples, 3)),
        dirs=jnp.zeros((max_samples, 3)),
        dts=jnp.zeros((max_samples,)),
        valid=jnp.zeros((max_samples,), dtype=bool),
        n_iter=jnp.zeros((), jnp.int32),
    )

    def cond_fn(c: _Carry) -> Bool[Array, ""]:
        return (c.t < t1) & (c.n_valid < max_samples) & (c.n_iter < max_march_iters)

    def body_fn(c: _Carry) -> _Carry:
        pos = ray_o + c.t * ray_d
        mip = mip_from_pos(pos, grid.n_cascades - 1)
        occ = is_occupied(grid, pos, mip)
        dt = cone_angle_min_stepsize * (2.0 ** mip.astype(jnp.float32))

        def on_sample(c: _Carry) -> _Carry:
            idx = c.n_valid
            return _Carry(
                t=c.t + dt,
                n_valid=c.n_valid + 1,
                positions=c.positions.at[idx].set(pos),
                dirs=c.dirs.at[idx].set(ray_d),
                dts=c.dts.at[idx].set(dt),
                valid=c.valid.at[idx].set(True),
                n_iter=c.n_iter + 1,
            )

        def on_skip(c: _Carry) -> _Carry:
            t_next = c.t + _distance_to_next_voxel(pos, ray_d, mip, grid.grid_size)
            return _Carry(
                t=t_next,
                n_valid=c.n_valid,
                positions=c.positions,
                dirs=c.dirs,
                dts=c.dts,
                valid=c.valid,
                n_iter=c.n_iter + 1,
            )

        return jax.lax.cond(occ, on_sample, on_skip, c)

    final = jax.lax.while_loop(cond_fn, body_fn, init)
    return MarchResult(final.positions, final.dirs, final.dts, final.valid, final.n_valid)


def march_rays(
    rays_o: Float[Array, "n_rays 3"],
    rays_d: Float[Array, "n_rays 3"],
    grid: OccupancyGrid,
    aabb: BoundingBox,
    max_samples: int,
    max_march_iters: int,
    cone_angle_min_stepsize: float,
    near_distance: float,
) -> MarchResult:
    return jax.vmap(march_ray, in_axes=(0, 0, None, None, None, None, None, None))(
        rays_o, rays_d, grid, aabb, max_samples, max_march_iters, cone_angle_min_stepsize, near_distance
    )
