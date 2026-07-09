"""Sphere-trace + shade a trained SdfNetwork checkpoint to PNG."""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from jaxingp.config import SdfNetworkConfig
from jaxingp.geometry.aabb import BoundingBox
from jaxingp.geometry.rays import orbit_c2w, orbit_camera_rays
from jaxingp.nn.sdf_network import SdfNetwork
from jaxingp.render.sphere_trace import render_rays_sdf
from jaxingp.training import checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--azimuth-deg", type=float, default=35.0)
    parser.add_argument("--elevation-deg", type=float, default=15.0)
    parser.add_argument("--out", type=str, default="/tmp/render_sdf.png")
    args = parser.parse_args()

    key = jax.random.PRNGKey(0)
    model = checkpoint.load(args.checkpoint, SdfNetwork(key, SdfNetworkConfig()))

    aabb = BoundingBox(min_corner=-jnp.ones(3), max_corner=jnp.ones(3))
    c2w = orbit_c2w(jnp.zeros(3), args.radius, args.azimuth_deg, args.elevation_deg)
    fov_x = jnp.radians(40.0)
    fx = args.width / (2.0 * jnp.tan(fov_x / 2.0))
    cx, cy = args.width / 2.0, args.height / 2.0
    rays_o, rays_d = orbit_camera_rays(c2w, fx, fx, cx, cy, args.height, args.width)

    light_dir = jnp.array([0.5, 0.7, 0.5])
    light_dir = light_dir / jnp.linalg.norm(light_dir)
    background = jnp.ones(3)

    rgb = render_rays_sdf(model, aabb, rays_o, rays_d, light_dir, background)
    img = np.asarray(jnp.clip(rgb.reshape(args.height, args.width, 3), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
