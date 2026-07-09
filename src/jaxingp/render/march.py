import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from jaxingp.geometry.aabb import BoundingBox
from jaxingp.occupancy.grid import OccupancyGrid, is_occupied, mip_from_dt, mip_from_pos

# Fixed physics-like constants of the marching scheme itself (nerf_device.cuh:29-36),
# independent of any per-scene parameter (aabb_scale, actual grid_size used, etc.):
# NERF_GRIDSIZE=128, NERF_CASCADES=8, NERF_STEPS=1024, STEPSIZE=sqrt(3)/NERF_STEPS.
_SQRT3 = math.sqrt(3.0)
MIN_CONE_STEPSIZE = _SQRT3 / 1024.0
MAX_CONE_STEPSIZE = MIN_CONE_STEPSIZE * (1 << 7) * 1024 / 128


def to_stepping_space(t: Float[Array, ""], cone_angle: float) -> Float[Array, ""]:
    """Ports instant-ngp's `to_stepping_space` (nerf_device.cuh:376-393)
    exactly. `cone_angle` is always a plain Python float in this codebase
    (derived once per dataset from aabb_scale), so the cone_angle<=1e-5
    branch is a real Python `if` — not a traced `jnp.where` — since the
    general branch below divides by `log1p_c`, which is exactly 0 when
    cone_angle=0 and would produce NaNs if both branches were evaluated."""
    if cone_angle <= 1e-5:
        return t / MIN_CONE_STEPSIZE

    log1p_c = math.log(1.0 + cone_angle)
    a = (math.log(MIN_CONE_STEPSIZE) - math.log(log1p_c)) / log1p_c
    b = (math.log(MAX_CONE_STEPSIZE) - math.log(log1p_c)) / log1p_c
    at = math.exp(a * log1p_c)
    bt = math.exp(b * log1p_c)

    return jnp.where(
        t <= at,
        (t - at) / MIN_CONE_STEPSIZE + a,
        jnp.where(t <= bt, jnp.log(t) / log1p_c, (t - bt) / MAX_CONE_STEPSIZE + b),
    )


def from_stepping_space(n: Float[Array, ""], cone_angle: float) -> Float[Array, ""]:
    """Ports instant-ngp's `from_stepping_space` (nerf_device.cuh:395-413) exactly."""
    if cone_angle <= 1e-5:
        return n * MIN_CONE_STEPSIZE

    log1p_c = math.log(1.0 + cone_angle)
    a = (math.log(MIN_CONE_STEPSIZE) - math.log(log1p_c)) / log1p_c
    b = (math.log(MAX_CONE_STEPSIZE) - math.log(log1p_c)) / log1p_c
    at = math.exp(a * log1p_c)
    bt = math.exp(b * log1p_c)

    return jnp.where(
        n <= a,
        (n - a) * MIN_CONE_STEPSIZE + at,
        jnp.where(n <= b, jnp.exp(n * log1p_c), (n - b) * MAX_CONE_STEPSIZE + bt),
    )


def calc_dt(t: Float[Array, ""], cone_angle: float) -> Float[Array, ""]:
    """Ports `calc_dt`/`advance_n_steps(t, cone_angle, 1)` (nerf_device.cuh:425-429).
    At cone_angle=0 (aabb_scale<=1) this degenerates to the constant
    MIN_CONE_STEPSIZE; otherwise dt grows continuously (log-space) with t."""
    return from_stepping_space(to_stepping_space(t, cone_angle) + 1.0, cone_angle) - t


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
    the cascaded grid at cascade `mip`, ported faithfully from CUDA's
    `distance_to_next_voxel` geometry (verified equivalent up to an
    integer-valued origin shift)."""
    scale = 2.0 ** (-mip.astype(jnp.float32))
    p_scaled = ((pos - 0.5) * scale + 0.5) * grid_size
    d_scaled = ray_d * scale * grid_size

    t_pos = jnp.where(d_scaled > 0, (jnp.floor(p_scaled) + 1 - p_scaled) / d_scaled, jnp.inf)
    t_neg = jnp.where(d_scaled < 0, (jnp.ceil(p_scaled) - 1 - p_scaled) / d_scaled, jnp.inf)
    return jnp.maximum(jnp.min(jnp.minimum(t_pos, t_neg)), 0.0)


def march_ray(
    ray_o: Float[Array, "3"],
    ray_d: Float[Array, "3"],
    grid: OccupancyGrid,
    aabb: BoundingBox,
    max_samples: int,
    max_march_iters: int,
    cone_angle: float,
    max_cascade: int,
    near_distance: float,
    use_mip_from_dt: bool,
) -> MarchResult:
    """Per-ray adaptive marcher: `jax.lax.while_loop`, vmapped externally
    across rays (see march_rays). Each ray owns a fixed-size padded output
    slot (positions/dirs/dts/valid of length max_samples) rather than a
    globally-compacted buffer — see PLAN.md "Ray marching" section for why
    this is the JAX-native substitute for CUDA's atomicAdd-compaction scheme.

    `dt` and the skip-jump quantization now faithfully port CUDA's
    log-space `calc_dt`/`advance_to_next_voxel` (nerf_device.cuh:360-439)
    rather than the earlier discrete `dt = const * 2**mip` simplification.
    `use_mip_from_dt` selects CUDA's training-path cascade lookup
    (`mip_from_dt`, dt-aware) vs. its render-path lookup (`mip_from_pos`,
    position-only) — see occupancy/grid.py.
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
        dt = calc_dt(c.t, cone_angle)
        if use_mip_from_dt:
            mip = mip_from_dt(dt, pos, max_cascade, grid.grid_size)
        else:
            mip = mip_from_pos(pos, max_cascade)
        occ = is_occupied(grid, pos, mip)

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
            dist = _distance_to_next_voxel(pos, ray_d, mip, grid.grid_size)
            t_step = to_stepping_space(c.t, cone_angle)
            t_target_step = to_stepping_space(c.t + dist, cone_angle)
            t_next = from_stepping_space(
                t_step + jnp.ceil(jnp.maximum(t_target_step - t_step, 0.5)), cone_angle
            )
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
    cone_angle: float,
    max_cascade: int,
    near_distance: float,
    use_mip_from_dt: bool = False,
) -> MarchResult:
    return jax.vmap(march_ray, in_axes=(0, 0, None, None, None, None, None, None, None, None))(
        rays_o, rays_d, grid, aabb, max_samples, max_march_iters, cone_angle, max_cascade,
        near_distance, use_mip_from_dt,
    )
