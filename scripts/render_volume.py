"""Phase 7 validation: no training involved — synthesize (or load) a raw
voxel grid, ray-march it with trilinear sampling, and save a render. Visual
correctness check only (does the renderer reproduce the input volume)."""

import argparse

import jax.numpy as jnp
import numpy as np
from PIL import Image

from jaxingp.data.voxel_dataset import load_voxel_grid, synthesize_toy_volume
from jaxingp.geometry.aabb import BoundingBox
from jaxingp.geometry.rays import orbit_c2w, orbit_camera_rays
from jaxingp.render.volume_march import render_rays_volume


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxel-path", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=64, help="toy volume resolution if --voxel-path unset")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--azimuth-deg", type=float, default=35.0)
    parser.add_argument("--elevation-deg", type=float, default=20.0)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--out", type=str, default="/tmp/render_volume.png")
    args = parser.parse_args()

    volume = load_voxel_grid(args.voxel_path) if args.voxel_path else synthesize_toy_volume(args.resolution)
    print(f"volume shape {volume.shape}")

    center = jnp.array([0.5, 0.5, 0.5])
    c2w = orbit_c2w(center, args.radius, args.azimuth_deg, args.elevation_deg)

    fov_x = jnp.radians(50.0)
    fx = args.width / (2.0 * jnp.tan(fov_x / 2.0))
    fy = fx
    cx, cy = args.width / 2.0, args.height / 2.0

    rays_o, rays_d = orbit_camera_rays(c2w, fx, fy, cx, cy, args.height, args.width)

    aabb = BoundingBox()
    background = jnp.ones(3)
    rgb = render_rays_volume(volume, aabb, rays_o, rays_d, args.n_samples, background)

    img = np.asarray(jnp.clip(rgb.reshape(args.height, args.width, 3), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
