"""Interactive web-based viewer for trained NeRF/SDF/Volume checkpoints,
using viser (WebSocket + browser, orbit camera controls). Re-renders from
the trained JAX model live as the camera moves.

    uv run python scripts/viewer.py nerf --checkpoint checkpoints/nerf_adaptive_full
    uv run python scripts/viewer.py sdf --checkpoint checkpoints/sdf/model.eqx
    uv run python scripts/viewer.py volume --voxel-path path/to/volume.npy

Then open the printed http://localhost:8080 URL in a browser.

Viser's camera frame is OpenCV-style (local +Z forward, +Y down) —
confirmed empirically (see PLAN.md/session notes): the OpenGL convention
(-Z forward, +Y up) used everywhere else in this codebase pointed every
ray away from the scene regardless of orbit direction. `viser_camera_rays`
below is deliberately separate from geometry/rays.py's `orbit_camera_rays`
to keep that convention out of the rest of the codebase, which does use
OpenGL/NeRF convention throughout (matches instant-ngp's transform_matrix
convention).
"""

import argparse
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import viser
import viser.transforms as vtf

from jaxingp.config import NerfNetworkConfig, SdfNetworkConfig
from jaxingp.data.voxel_dataset import load_voxel_grid, synthesize_toy_volume
from jaxingp.geometry.aabb import BoundingBox
from jaxingp.nn.nerf_network import NerfNetwork
from jaxingp.nn.sdf_network import SdfNetwork
from jaxingp.occupancy.grid import OccupancyGrid
from jaxingp.render.render import render_rays_adaptive, render_rays_uniform
from jaxingp.render.sphere_trace import render_rays_sdf
from jaxingp.render.volume_march import render_rays_volume
from jaxingp.training import checkpoint


def build_c2w(camera: viser.CameraHandle) -> jnp.ndarray:
    rot = jnp.array(vtf.SO3(camera.wxyz).as_matrix())
    return jnp.eye(4).at[:3, :3].set(rot).at[:3, 3].set(jnp.array(camera.position))


def viser_camera_rays(c2w, fx, fy, cx, cy, h, w):
    """OpenCV-style camera rays (+Z forward, +Y down) matching viser's
    camera frame — see module docstring."""
    ys, xs = jnp.meshgrid(
        jnp.arange(h, dtype=jnp.float32), jnp.arange(w, dtype=jnp.float32), indexing="ij"
    )
    px, py = xs.reshape(-1), ys.reshape(-1)
    dir_cam = jnp.stack([(px - cx) / fx, (py - cy) / fy, jnp.ones_like(px)], axis=-1)
    d_world = dir_cam @ c2w[:3, :3].T
    d_world = d_world / jnp.linalg.norm(d_world, axis=-1, keepdims=True)
    o_world = jnp.broadcast_to(c2w[:3, 3], d_world.shape)
    return o_world, d_world


def make_render_fn(args):
    """Returns a jitted (rays_o, rays_d) -> rgb_flat function for the
    selected primitive. Deliberately not chunked (unlike the eval-time
    render_*_chunked helpers used during training) — the viewer's
    resolution is small enough to render in one batch, and staying
    unchunked means only one jit compilation instead of one per chunk."""
    background = jnp.ones(3)

    if args.mode == "nerf":
        key = jax.random.PRNGKey(0)
        model = checkpoint.load(
            os.path.join(args.checkpoint, "ema_model.eqx"), NerfNetwork(key, NerfNetworkConfig())
        )
        aabb = BoundingBox()
        grid_path = os.path.join(args.checkpoint, "grid.eqx")

        if os.path.exists(grid_path):
            grid = checkpoint.load(
                grid_path, OccupancyGrid(grid_size=args.grid_size, n_cascades=args.n_cascades)
            )
            march_cfg = (args.max_samples, args.max_march_iters, args.cone_min_stepsize, args.near_distance)

            @eqx.filter_jit
            def render(rays_o, rays_d):
                rgb, _, _ = render_rays_adaptive(model, grid, aabb, rays_o, rays_d, *march_cfg, background)
                return rgb
        else:
            @eqx.filter_jit
            def render(rays_o, rays_d):
                rgb, _ = render_rays_uniform(
                    model, aabb, rays_o, rays_d, args.n_samples, jax.random.PRNGKey(0), background
                )
                return rgb

        return render, (0.5, 0.5, 0.5), 2.0

    if args.mode == "sdf":
        key = jax.random.PRNGKey(0)
        model = checkpoint.load(args.checkpoint, SdfNetwork(key, SdfNetworkConfig()))
        aabb = BoundingBox(min_corner=-jnp.ones(3), max_corner=jnp.ones(3))
        light_dir = jnp.array([0.5, 0.7, 0.5])
        light_dir = light_dir / jnp.linalg.norm(light_dir)

        @eqx.filter_jit
        def render(rays_o, rays_d):
            return render_rays_sdf(model, aabb, rays_o, rays_d, light_dir, background)

        return render, (0.0, 0.0, 0.0), 3.0

    if args.mode == "volume":
        volume = load_voxel_grid(args.voxel_path) if args.voxel_path else synthesize_toy_volume(args.resolution)
        aabb = BoundingBox()

        @eqx.filter_jit
        def render(rays_o, rays_d):
            return render_rays_volume(volume, aabb, rays_o, rays_d, args.n_samples, background)

        return render, (0.5, 0.5, 0.5), 2.0

    raise ValueError(args.mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["nerf", "sdf", "volume"])
    parser.add_argument("--checkpoint", type=str, help="checkpoint dir (nerf) or model.eqx path (sdf)")
    parser.add_argument("--voxel-path", type=str, default=None, help="volume mode: .npy RGB+density array")
    parser.add_argument("--resolution", type=int, default=64, help="volume mode: toy-volume resolution if unset")
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-march-iters", type=int, default=1024)
    parser.add_argument("--cone-min-stepsize", type=float, default=1.0 / 1024)
    parser.add_argument("--near-distance", type=float, default=1e-3)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--n-cascades", type=int, default=8)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    render_fn, scene_center, scene_radius = make_render_fn(args)
    width, height = args.width, args.height

    def render_and_update(client: viser.ClientHandle) -> None:
        try:
            c2w = build_c2w(client.camera)
            fy = height / (2.0 * jnp.tan(client.camera.fov / 2.0))
            rays_o, rays_d = viser_camera_rays(c2w, fy, fy, width / 2.0, height / 2.0, height, width)
            rgb = render_fn(rays_o, rays_d)
            img = np.asarray(jnp.clip(rgb, 0, 1).reshape(height, width, 3) * 255).astype(np.uint8)
            client.scene.set_background_image(img, format="jpeg")
        except Exception:
            import traceback

            traceback.print_exc()

    server = viser.ViserServer(port=args.port)
    print(f"viewer running — open the URL above in a browser (mode={args.mode})")

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # Viser's default camera starts tens of units away, meant for
        # real-world-scale scenes; ours live in a ~1-2 unit box, so the
        # scene would be an invisible speck without repositioning.
        client.camera.position = tuple(c + scene_radius for c in scene_center)
        client.camera.look_at = scene_center

        @client.camera.on_update
        def _(_) -> None:
            render_and_update(client)

        render_and_update(client)

    server.sleep_forever()


if __name__ == "__main__":
    main()
