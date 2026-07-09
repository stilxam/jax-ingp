from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray


class OccupancyGrid(eqx.Module):
    """Cascaded density occupancy grid (EMA state, not gradient-descended).

    Dense array, not a packed Morton bitfield: Z-order bit-packing in the
    CUDA original exists for warp memory coherence, which has no XLA
    analogue. A dense (n_cascades, res, res, res) float32 array is trivially
    affordable (~4MB at res=64, ~67MB at res=128) and simpler to gather from.
    """

    density: Float[Array, "n_cascades res res res"]
    mean_density: Float[Array, ""]

    grid_size: int = eqx.field(static=True)
    n_cascades: int = eqx.field(static=True)
    decay: float = eqx.field(static=True)
    threshold_const: float = eqx.field(static=True)

    def __init__(
        self,
        grid_size: int = 128,
        n_cascades: int = 8,
        decay: float = 0.95,
        threshold_const: float = 0.01,
    ):
        self.grid_size = grid_size
        self.n_cascades = n_cascades
        self.decay = decay
        self.threshold_const = threshold_const
        self.density = jnp.zeros((n_cascades, grid_size, grid_size, grid_size))
        self.mean_density = jnp.zeros(())


def mip_from_pos(pos: Float[Array, "3"], max_cascade: int) -> Int[Array, ""]:
    """Cascade index covering `pos` (unit-cube coords). Approximates the
    frexp-based `mip_from_pos` in nerf_device.cuh: cascade 0 covers [0,1]^3
    (half-extent 0.5 around the 0.5 center), cascade m covers half-extent
    0.5 * 2^m."""
    maxcomp = jnp.max(jnp.abs(pos - 0.5))
    exponent = jnp.floor(jnp.log2(jnp.maximum(maxcomp, 1e-10)))
    return jnp.clip(exponent + 2.0, 0, max_cascade).astype(jnp.int32)


def cascaded_grid_coord(
    pos: Float[Array, "3"], mip: Int[Array, ""], grid_size: int
) -> tuple[Int[Array, "3"], Bool[Array, ""]]:
    scale = 2.0 ** (-mip.astype(jnp.float32))
    p = (pos - 0.5) * scale + 0.5
    idx = jnp.floor(p * grid_size).astype(jnp.int32)
    in_bounds = jnp.all((idx >= 0) & (idx < grid_size))
    return idx, in_bounds


def pos_from_cascaded_grid_coord(
    idx: Float[Array, "3"], mip: Int[Array, ""], grid_size: int
) -> Float[Array, "3"]:
    scale = 2.0 ** mip.astype(jnp.float32)
    p = (idx + 0.5) / grid_size
    return (p - 0.5) * scale + 0.5


def is_occupied(grid: OccupancyGrid, pos: Float[Array, "3"], mip: Int[Array, ""]) -> Bool[Array, ""]:
    """Flat density threshold — matches instant-ngp's `grid_to_bitfield`
    (testbed_nerf.cu:348) exactly: `thresh = min(NERF_MIN_OPTICAL_THICKNESS,
    mean_density)`, then `density > thresh`, with no `dt` in the comparison
    despite the constant's name. (An earlier version of this function
    divided by `dt` to make this a literal optical-thickness test — that
    was based on a plausible-sounding misreading of the constant's name,
    not the actual CUDA source; reverted after checking.)"""
    idx, in_bounds = cascaded_grid_coord(pos, mip, grid.grid_size)
    idx = jnp.clip(idx, 0, grid.grid_size - 1)
    thresh = jnp.minimum(grid.threshold_const, grid.mean_density)
    d = grid.density[mip, idx[0], idx[1], idx[2]]
    return in_bounds & (d > thresh)


def update_occupancy_grid(
    grid: OccupancyGrid,
    density_fn: Callable[[Float[Array, "3"]], Float[Array, ""]],
    key: PRNGKeyArray,
) -> OccupancyGrid:
    """Evaluates density over the ENTIRE grid (all n_cascades * res^3 cells,
    jittered cell centers) in one batched vmapped pass. Simplification vs.
    CUDA's stochastic per-call subsampling: a full sweep through the tiny
    density MLP is a few ms on GPU, removing the need to replicate CUDA's
    subsample scheduling. `new_density = max(density * decay, evaluated)`
    (EMA-as-max). Cells permanently pruned by `mark_untrained_density_grid`
    (negative sentinel, never seen by any training camera) are left
    untouched — they never rejoin the EMA blend, matching CUDA's behavior
    of never re-evaluating cells no camera can ever supervise."""
    res, n_cascades = grid.grid_size, grid.n_cascades
    ii, jj, kk = jnp.meshgrid(
        jnp.arange(res), jnp.arange(res), jnp.arange(res), indexing="ij"
    )
    idx = jnp.stack([ii, jj, kk], axis=-1).astype(jnp.float32)  # (res,res,res,3)
    jitter = jax.random.uniform(key, idx.shape, minval=0.0, maxval=1.0)
    idx = idx + jitter
    idx_flat = idx.reshape(-1, 3)

    def eval_mip(mip):
        positions = jax.vmap(pos_from_cascaded_grid_coord, in_axes=(0, None, None))(
            idx_flat, mip, res
        )
        return jax.vmap(density_fn)(positions).reshape(res, res, res)

    new_eval = jax.vmap(eval_mip)(jnp.arange(n_cascades))  # (n_cascades,res,res,res)
    permanently_pruned = grid.density < 0
    new_density = jnp.where(
        permanently_pruned, grid.density, jnp.maximum(grid.density * grid.decay, new_eval)
    )
    live = ~permanently_pruned
    mean_density = jnp.sum(jnp.where(live, new_density, 0.0)) / jnp.maximum(jnp.sum(live), 1)

    return eqx.tree_at(
        lambda g: (g.density, g.mean_density), grid, (new_density, mean_density)
    )


_VISIBLE_SEED_DENSITY = 0.0
_INVISIBLE_SENTINEL = -1.0


def mark_untrained_density_grid(
    grid: OccupancyGrid,
    c2w_world: Float[Array, "n_cameras 4 4"],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    w: float,
    h: float,
    scale: Float[Array, ""],
    offset: Float[Array, "3"],
    chunk_size: int = 200_000,
) -> OccupancyGrid:
    """Bootstraps the grid before any training, matching CUDA's
    `mark_untrained_density_grid` (testbed_nerf.cu:87) exactly: cells
    inside at least one training camera's view frustum are seeded to
    density=0.0 (not a large "guaranteed occupied" value — that was an
    earlier, incorrect guess; the real seed is just 0.0, immediately
    overwritten by a real evaluation on the very next grid update, which
    CUDA folds into the same step-0 call — see `training/train_nerf.py`);
    cells no camera can ever see are seeded with a negative sentinel and
    permanently pruned (`update_occupancy_grid` never re-blends them)."""
    cam_pos_unit = c2w_world[:, :3, 3] * scale + offset  # uniform scale preserves angles/FOV
    cam_rot = c2w_world[:, :3, :3]

    def visible_by_camera(pos: Float[Array, "3"], rot: Float[Array, "3 3"], cam_pos: Float[Array, "3"]) -> Bool[Array, ""]:
        p_cam = rot.T @ (pos - cam_pos)
        t = -p_cam[2]  # camera looks down -z (see geometry/rays.py convention)
        xu = p_cam[0] / jnp.where(t != 0, t, 1e-8)
        yu = -p_cam[1] / jnp.where(t != 0, t, 1e-8)
        px = xu * fx + cx
        py = yu * fy + cy
        return (t > 0) & (px >= 0) & (px < w) & (py >= 0) & (py < h)

    def visible_by_any_camera(pos: Float[Array, "3"]) -> Bool[Array, ""]:
        per_cam = jax.vmap(visible_by_camera, in_axes=(None, 0, 0))(pos, cam_rot, cam_pos_unit)
        return jnp.any(per_cam)

    res, n_cascades = grid.grid_size, grid.n_cascades
    ii, jj, kk = jnp.meshgrid(jnp.arange(res), jnp.arange(res), jnp.arange(res), indexing="ij")
    idx_flat = (jnp.stack([ii, jj, kk], axis=-1).astype(jnp.float32) + 0.5).reshape(-1, 3)

    @jax.jit
    def visible_chunk(positions_chunk):
        return jax.vmap(visible_by_any_camera)(positions_chunk)

    # Chunked over cells (not just vmapped over all n_cascades*res^3 at once):
    # a single unchunked vmap over e.g. 8*128^3 cells x 50 cameras materializes
    # an intermediate on the order of cells*cameras*3 floats (~10GB at
    # grid_size=128), which OOMs even on a 24GB GPU. This is a one-time
    # startup cost (not the hot training loop), so a plain Python chunking
    # loop is fine — only ~2 distinct chunk shapes ever get compiled.
    visible_per_mip = []
    for mip in range(n_cascades):
        positions = jax.vmap(pos_from_cascaded_grid_coord, in_axes=(0, None, None))(
            idx_flat, jnp.asarray(mip), res
        )
        chunks = [
            visible_chunk(positions[start : start + chunk_size])
            for start in range(0, positions.shape[0], chunk_size)
        ]
        visible_per_mip.append(jnp.concatenate(chunks).reshape(res, res, res))
    visible = jnp.stack(visible_per_mip, axis=0)  # (n_cascades,res,res,res)
    seeded_density = jnp.where(visible, _VISIBLE_SEED_DENSITY, _INVISIBLE_SENTINEL)
    mean_density = jnp.sum(jnp.where(visible, seeded_density, 0.0)) / jnp.maximum(jnp.sum(visible), 1)

    return eqx.tree_at(
        lambda g: (g.density, g.mean_density), grid, (seeded_density, mean_density)
    )
