import argparse
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from tqdm import tqdm

from jaxingp.config import NerfNetworkConfig
from jaxingp.data.nerf_dataset import NerfDataset
from jaxingp.geometry.aabb import BoundingBox
from jaxingp.nn.nerf_network import NerfNetwork
from jaxingp.occupancy.grid import OccupancyGrid, mark_untrained_density_grid, update_occupancy_grid
from jaxingp.render.render import (
    render_rays_adaptive,
    render_rays_adaptive_chunked,
    render_rays_uniform,
    render_rays_uniform_chunked,
)
from jaxingp.training import checkpoint
from jaxingp.training.logging_utils import setup_logger
from jaxingp.training.optim import build_nerf_optimizer, ema_update


def psnr(mse: jnp.ndarray) -> jnp.ndarray:
    return -10.0 * jnp.log10(jnp.maximum(mse, 1e-10))


def loss_uniform(model, aabb, rays_o, rays_d, target, n_samples, key, background):
    pred_rgb, _ = render_rays_uniform(model, aabb, rays_o, rays_d, n_samples, key, background)
    return jnp.mean((pred_rgb - target) ** 2)


def loss_adaptive(model, grid, aabb, rays_o, rays_d, target, march_cfg, background):
    pred_rgb, _, n_valid = render_rays_adaptive(
        model, grid, aabb, rays_o, rays_d, *march_cfg, background
    )
    return jnp.mean((pred_rgb - target) ** 2), n_valid


def _step_uniform(model, opt_state, aabb, rays_o, rays_d, target, n_samples, key, background, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_uniform)(
        model, aabb, rays_o, rays_d, target, n_samples, key, background
    )
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def _step_adaptive(model, opt_state, grid, aabb, rays_o, rays_d, target, march_cfg, background, optimizer):
    (loss, n_valid), grads = eqx.filter_value_and_grad(loss_adaptive, has_aux=True)(
        model, grid, aabb, rays_o, rays_d, target, march_cfg, background
    )
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, n_valid


def make_uniform_chunk_fn(dataset, aabb, n_samples, batch_size, held_out_idx, ema_decay, optimizer):
    """One `lax.scan`-compiled chunk of `chunk_len` uniform-sampling training
    steps. `dataset.sample_batch` is pure JAX on already-resident device
    arrays, so the whole step is traceable — scanning many steps together
    means only one host<->device sync per chunk instead of one per step,
    eliminating per-step Python dispatch overhead."""

    def step_fn(static, carry, _):
        model_arr, opt_state, ema_arr, key = carry
        model = eqx.combine(model_arr, static)
        ema_model = eqx.combine(ema_arr, static)

        key, sample_key, render_key, bg_key = jax.random.split(key, 4)
        rays_o, rays_d, target = dataset.sample_batch(sample_key, batch_size, exclude_idx=held_out_idx)
        train_background = jax.random.uniform(bg_key, (3,))
        model, opt_state, loss = _step_uniform(
            model, opt_state, aabb, rays_o, rays_d, target, n_samples, render_key, train_background, optimizer
        )
        ema_model = ema_update(ema_model, model, ema_decay)

        model_arr, _ = eqx.partition(model, eqx.is_array)
        ema_arr, _ = eqx.partition(ema_model, eqx.is_array)
        return (model_arr, opt_state, ema_arr, key), loss

    @eqx.filter_jit
    def run_chunk(model, opt_state, ema_model, key, chunk_len):
        # lax.scan's carry must be pure arrays — model/ema_model carry non-array
        # leaves too (e.g. activation functions), so partition once here and
        # close over the (unchanging) static skeleton inside step_fn instead
        # of threading it through the carry.
        model_arr, static = eqx.partition(model, eqx.is_array)
        ema_arr, _ = eqx.partition(ema_model, eqx.is_array)
        carry, losses = jax.lax.scan(
            lambda c, x: step_fn(static, c, x), (model_arr, opt_state, ema_arr, key), xs=None, length=chunk_len
        )
        model_arr, opt_state, ema_arr, key = carry
        return (eqx.combine(model_arr, static), opt_state, eqx.combine(ema_arr, static), key), losses

    return run_chunk


def make_adaptive_chunk_fn(dataset, aabb, march_cfg, batch_size, held_out_idx, ema_decay, optimizer, grid_update_interval):
    # NerfNetwork carries non-array leaves (e.g. eqx.nn.MLP's activation
    # function), which lax.scan's carry can't hold — partition/combine
    # around model & ema_model, same as make_uniform_chunk_fn. OccupancyGrid
    # doesn't need this: its non-array fields are already eqx.field(static=True),
    # so equinox excludes them from its pytree leaves entirely.
    def step_fn(static, carry, _):
        model_arr, opt_state, ema_arr, grid, key, step = carry
        model = eqx.combine(model_arr, static)
        ema_model = eqx.combine(ema_arr, static)

        key, sample_key, render_key, bg_key, grid_key = jax.random.split(key, 5)
        rays_o, rays_d, target = dataset.sample_batch(sample_key, batch_size, exclude_idx=held_out_idx)
        train_background = jax.random.uniform(bg_key, (3,))

        # Matches CUDA's training_prep_nerf cadence (testbed_nerf.cu:3392):
        # every step for the first 256, then every `grid_update_interval`
        # steps — not the coarse fixed interval from step 0 this used to
        # use, which let the grid go stale for stretches during exactly
        # the phase where the network's density predictions are changing
        # fastest.
        should_update = (step < 256) | (step % grid_update_interval == 0)
        grid = jax.lax.cond(
            should_update,
            lambda g: update_occupancy_grid(g, model.density, grid_key),
            lambda g: g,
            grid,
        )

        model, opt_state, loss, n_valid = _step_adaptive(
            model, opt_state, grid, aabb, rays_o, rays_d, target, march_cfg, train_background, optimizer
        )
        ema_model = ema_update(ema_model, model, ema_decay)

        model_arr, _ = eqx.partition(model, eqx.is_array)
        ema_arr, _ = eqx.partition(ema_model, eqx.is_array)
        return (model_arr, opt_state, ema_arr, grid, key, step + 1), (loss, n_valid)

    @eqx.filter_jit
    def run_chunk(model, opt_state, ema_model, grid, key, step, chunk_len):
        model_arr, static = eqx.partition(model, eqx.is_array)
        ema_arr, _ = eqx.partition(ema_model, eqx.is_array)
        carry, (losses, n_valids) = jax.lax.scan(
            lambda c, x: step_fn(static, c, x),
            (model_arr, opt_state, ema_arr, grid, key, step),
            xs=None, length=chunk_len,
        )
        model_arr, opt_state, ema_arr, grid, key, step = carry
        return (eqx.combine(model_arr, static), opt_state, eqx.combine(ema_arr, static), grid, key, step), losses, n_valids

    return run_chunk


def save_image(arr: jnp.ndarray, h: int, w: int, path: str) -> None:
    img = np.asarray(jnp.clip(arr.reshape(h, w, 3), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def save_checkpoint(out_dir, model, ema_model, opt_state, grid, step):
    checkpoint.save(os.path.join(out_dir, "model.eqx"), model)
    checkpoint.save(os.path.join(out_dir, "ema_model.eqx"), ema_model)
    checkpoint.save(os.path.join(out_dir, "opt_state.eqx"), opt_state)
    if grid is not None:
        checkpoint.save(os.path.join(out_dir, "grid.eqx"), grid)
    with open(os.path.join(out_dir, "step.txt"), "w") as f:
        f.write(str(step))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("transforms", type=str)
    parser.add_argument("--marcher", choices=["uniform", "adaptive"], default="uniform")
    parser.add_argument("--downscale", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=64, help="adaptive marcher: max samples/ray")
    parser.add_argument("--max-march-iters", type=int, default=1024)
    parser.add_argument("--cone-min-stepsize", type=float, default=1.0 / 1024)
    parser.add_argument("--near-distance", type=float, default=1e-3)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--n-cascades", type=int, default=8)
    parser.add_argument("--grid-update-interval", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--decay-start", type=int, default=20000)
    parser.add_argument("--decay-interval", type=int, default=10000)
    parser.add_argument("--decay-base", type=float, default=0.33)
    parser.add_argument("--ema-decay", type=float, default=0.95)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--held-out-idx", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="checkpoints/nerf")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log = setup_logger(args.out_dir)

    dataset = NerfDataset.load(args.transforms, downscale=args.downscale)
    log.info(f"loaded {len(dataset.c2w)} frames at {dataset.w}x{dataset.h}, marcher={args.marcher}")

    key = jax.random.PRNGKey(0)
    model_key, data_key = jax.random.split(key)
    model = NerfNetwork(model_key, NerfNetworkConfig())
    ema_model = model
    aabb = BoundingBox()
    eval_background = jnp.ones(3)

    optimizer = build_nerf_optimizer(
        lr=args.lr, decay_start=args.decay_start, decay_interval=args.decay_interval, decay_base=args.decay_base
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    march_cfg = (args.max_samples, args.max_march_iters, args.cone_min_stepsize, args.near_distance)
    grid = None
    if args.marcher == "adaptive":
        grid = OccupancyGrid(grid_size=args.grid_size, n_cascades=args.n_cascades)
        grid = mark_untrained_density_grid(
            grid, dataset.c2w, dataset.fx, dataset.fy, dataset.cx, dataset.cy, dataset.w, dataset.h,
            dataset.scale, dataset.offset,
        )
        # The training loop's per-step `should_update` check (step < 256)
        # updates the grid before step 0's gradient step too, matching
        # CUDA's training_prep_nerf (called before every step including 0,
        # testbed_nerf.cu:3392-3397) — no separate pre-loop update needed.

    start_step = 0
    if args.resume:
        model = checkpoint.load(os.path.join(args.out_dir, "model.eqx"), model)
        ema_model = checkpoint.load(os.path.join(args.out_dir, "ema_model.eqx"), ema_model)
        opt_state = checkpoint.load(os.path.join(args.out_dir, "opt_state.eqx"), opt_state)
        if grid is not None:
            grid = checkpoint.load(os.path.join(args.out_dir, "grid.eqx"), grid)
        with open(os.path.join(args.out_dir, "step.txt")) as f:
            start_step = int(f.read())
        log.info(f"resumed from step {start_step}")

    if args.marcher == "uniform":
        run_chunk = make_uniform_chunk_fn(
            dataset, aabb, args.n_samples, args.batch_size, args.held_out_idx, args.ema_decay, optimizer
        )
    else:
        run_chunk = make_adaptive_chunk_fn(
            dataset, aabb, march_cfg, args.batch_size, args.held_out_idx, args.ema_decay, optimizer,
            args.grid_update_interval,
        )

    pbar = tqdm(total=args.steps, initial=start_step, desc="train")
    step = start_step
    while step < args.steps:
        chunk_len = min(args.eval_interval, args.steps - step)
        if args.marcher == "uniform":
            (model, opt_state, ema_model, data_key), losses = run_chunk(
                model, opt_state, ema_model, data_key, chunk_len
            )
        else:
            (model, opt_state, ema_model, grid, data_key, _), losses, n_valids = run_chunk(
                model, opt_state, ema_model, grid, data_key, jnp.asarray(step, jnp.int32), chunk_len
            )

        step += chunk_len
        pbar.update(chunk_len)
        loss = jnp.mean(losses)
        pbar.set_postfix(loss=f"{float(loss):.6f}")

        msg = f"step {step:5d}  loss {float(loss):.6f}"
        if args.marcher == "adaptive":
            msg += f"  n_valid mean {float(n_valids.mean()):.1f} max {int(n_valids.max())}"
        log.info(msg)

        rays_o, rays_d = dataset.render_rays_for_frame(args.held_out_idx)
        if args.marcher == "uniform":
            pred = render_rays_uniform_chunked(
                ema_model, aabb, rays_o, rays_d, args.n_samples, jax.random.PRNGKey(step), eval_background
            )
        else:
            pred, _ = render_rays_adaptive_chunked(
                ema_model, grid, aabb, rays_o, rays_d, *march_cfg, eval_background
            )
        gt = dataset.images[args.held_out_idx]
        mse = jnp.mean((pred.reshape(dataset.h, dataset.w, 3) - gt) ** 2)
        log.info(f"  held-out psnr {float(psnr(mse)):.2f}dB")
        save_image(pred, dataset.h, dataset.w, os.path.join(args.out_dir, f"step_{step:05d}.png"))
        save_checkpoint(args.out_dir, model, ema_model, opt_state, grid, step)

    log.info(f"saved checkpoint to {args.out_dir}")


if __name__ == "__main__":
    main()
