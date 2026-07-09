import argparse
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from tqdm import tqdm

from jaxingp.config import SdfNetworkConfig
from jaxingp.data.mesh_dataset import MeshDataset
from jaxingp.geometry.aabb import BoundingBox
from jaxingp.geometry.rays import orbit_c2w, orbit_camera_rays
from jaxingp.nn.sdf_network import SdfNetwork
from jaxingp.render.sphere_trace import render_rays_sdf
from jaxingp.training import checkpoint
from jaxingp.training.logging_utils import setup_logger
from jaxingp.training.optim import build_adam


def mape_loss(model: SdfNetwork, points: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    pred = jax.vmap(model)(points)
    return jnp.mean(jnp.abs(pred - target) / (jnp.abs(target) + 1e-2))


@eqx.filter_jit
def train_step(model, opt_state, points, target, optimizer):
    loss, grads = eqx.filter_value_and_grad(mape_loss)(model, points, target)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def render_preview(model: SdfNetwork, out_path: str, width: int, height: int, radius: float, azimuth_deg: float):
    aabb = BoundingBox(min_corner=-jnp.ones(3), max_corner=jnp.ones(3))
    c2w = orbit_c2w(jnp.zeros(3), radius, azimuth_deg, 15.0)
    fov_x = jnp.radians(40.0)
    fx = width / (2.0 * jnp.tan(fov_x / 2.0))
    cx, cy = width / 2.0, height / 2.0
    rays_o, rays_d = orbit_camera_rays(c2w, fx, fx, cx, cy, height, width)
    light_dir = jnp.array([0.5, 0.7, 0.5])
    light_dir = light_dir / jnp.linalg.norm(light_dir)
    background = jnp.ones(3)
    rgb = render_rays_sdf(model, aabb, rays_o, rays_d, light_dir, background)
    img = np.asarray(jnp.clip(rgb.reshape(height, width, 3), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=str)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--surface-frac", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--preview-width", type=int, default=256)
    parser.add_argument("--preview-height", type=int, default=256)
    parser.add_argument("--camera-radius", type=float, default=3.0)
    parser.add_argument("--out-dir", type=str, default="checkpoints/sdf")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log = setup_logger(args.out_dir)

    dataset = MeshDataset.load(args.mesh)
    log.info(f"loaded {args.mesh}: {len(dataset.mesh.vertices)} verts, {len(dataset.mesh.faces)} faces")

    key = jax.random.PRNGKey(0)
    model_key, data_key = jax.random.split(key)
    model = SdfNetwork(model_key, SdfNetworkConfig())
    optimizer = build_adam(lr=args.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    pbar = tqdm(range(args.steps), desc="train")
    for step in pbar:
        data_key, sample_key = jax.random.split(data_key)
        points, target = dataset.sample_batch(sample_key, args.batch_size, args.surface_frac)
        model, opt_state, loss = train_step(model, opt_state, points, target, optimizer)
        pbar.set_postfix(loss=f"{float(loss):.6f}")

        if step % args.eval_interval == 0 or step == args.steps - 1:
            log.info(f"step {step:5d}  MAPE loss {loss:.6f}")
            render_preview(
                model, os.path.join(args.out_dir, f"step_{step:05d}.png"),
                args.preview_width, args.preview_height, args.camera_radius, azimuth_deg=35.0,
            )

    checkpoint.save(os.path.join(args.out_dir, "model.eqx"), model)
    log.info(f"saved checkpoint to {args.out_dir}/model.eqx")


if __name__ == "__main__":
    main()
