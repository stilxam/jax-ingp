"""Render a held-out (or arbitrary dataset) frame from a trained NeRF
checkpoint to PNG, using the adaptive marcher if an occupancy grid
checkpoint is present, else falling back to uniform sampling."""

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from jaxingp.config import NerfNetworkConfig
from jaxingp.data.nerf_dataset import NerfDataset
from jaxingp.nn.nerf_network import NerfNetwork
from jaxingp.occupancy.grid import OccupancyGrid
from jaxingp.render.render import render_rays_adaptive_chunked, render_rays_uniform_chunked
from jaxingp.training import checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("transforms", type=str)
    parser.add_argument("checkpoint_dir", type=str)
    parser.add_argument("--downscale", type=int, default=8)
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-march-iters", type=int, default=1024)
    parser.add_argument("--near-distance", type=float, default=1e-3)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--n-cascades", type=int, default=8)
    parser.add_argument("--out", type=str, default="/tmp/novel_view.png")
    args = parser.parse_args()

    dataset = NerfDataset.load(args.transforms, downscale=args.downscale, n_cascades=args.n_cascades)
    aabb = dataset.aabb
    max_cascade = dataset.max_cascade
    cone_angle = 0.0 if max_cascade == 0 else 1.0 / 256.0
    background = jnp.ones(3)

    key = jax.random.PRNGKey(0)
    model = checkpoint.load(
        os.path.join(args.checkpoint_dir, "model.eqx"), NerfNetwork(key, NerfNetworkConfig())
    )

    rays_o, rays_d = dataset.render_rays_for_frame(args.frame_idx)

    grid_path = os.path.join(args.checkpoint_dir, "grid.eqx")
    if os.path.exists(grid_path):
        grid = checkpoint.load(grid_path, OccupancyGrid(grid_size=args.grid_size, n_cascades=args.n_cascades))
        pred, n_valid = render_rays_adaptive_chunked(
            model, grid, aabb, rays_o, rays_d,
            args.max_samples, args.max_march_iters, cone_angle, max_cascade, args.near_distance, background,
        )
        print(f"n_valid mean {float(n_valid.mean()):.1f} max {int(n_valid.max())}")
    else:
        pred = render_rays_uniform_chunked(
            model, aabb, rays_o, rays_d, args.n_samples, jax.random.PRNGKey(1), background
        )

    gt = dataset.images[args.frame_idx]
    mse = jnp.mean((pred.reshape(dataset.h, dataset.w, 3) - gt) ** 2)
    print(f"psnr {float(-10.0 * jnp.log10(jnp.maximum(mse, 1e-10))):.2f}dB")

    img = np.asarray(jnp.clip(pred.reshape(dataset.h, dataset.w, 3), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
