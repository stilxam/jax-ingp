"""Phase 4 validation: build an OccupancyGrid from a partially-trained NeRF
checkpoint, run one update, and visualize mid-height density slices per
cascade — should roughly outline the fox rather than being uniformly on/off.
"""

import argparse

import jax
import matplotlib.pyplot as plt

from jaxingp.config import NerfNetworkConfig
from jaxingp.nn.nerf_network import NerfNetwork
from jaxingp.occupancy.grid import OccupancyGrid, update_occupancy_grid
from jaxingp.training import checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--n-cascades", type=int, default=8)
    parser.add_argument("--out", type=str, default="/tmp/occupancy_slices.png")
    args = parser.parse_args()

    key = jax.random.PRNGKey(0)
    model_skeleton = NerfNetwork(key, NerfNetworkConfig())
    model = checkpoint.load(args.checkpoint, model_skeleton)

    grid = OccupancyGrid(grid_size=args.grid_size, n_cascades=args.n_cascades)
    grid = update_occupancy_grid(grid, model.density, jax.random.PRNGKey(1))
    grid = update_occupancy_grid(grid, model.density, jax.random.PRNGKey(2))

    print("density stats: min", float(grid.density.min()), "max", float(grid.density.max()))
    print("mean_density", float(grid.mean_density))
    thresh = min(grid.threshold_const, float(grid.mean_density))
    occ = grid.density > thresh
    print("occupied fraction per cascade:", [float(occ[m].mean()) for m in range(args.n_cascades)])

    fig, axes = plt.subplots(2, args.n_cascades, figsize=(3 * args.n_cascades, 6))
    mid = args.grid_size // 2
    for m in range(args.n_cascades):
        axes[0, m].imshow(grid.density[m, mid, :, :], cmap="viridis")
        axes[0, m].set_title(f"cascade {m} density")
        axes[0, m].axis("off")
        axes[1, m].imshow(occ[m, mid, :, :], cmap="gray")
        axes[1, m].set_title(f"cascade {m} occupied")
        axes[1, m].axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=100)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
