import argparse
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from tqdm import tqdm

from jaxingp.config import ImageNetworkConfig
from jaxingp.data.image_dataset import bilinear_sample, load_image, pixel_grid
from jaxingp.nn.image_network import ImageNetwork
from jaxingp.training import checkpoint
from jaxingp.training.logging_utils import setup_logger
from jaxingp.training.optim import build_adam


def psnr(mse: jnp.ndarray) -> jnp.ndarray:
    return -10.0 * jnp.log10(jnp.maximum(mse, 1e-10))


def loss_fn(model: ImageNetwork, xy: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    pred = jax.vmap(model)(xy)
    return jnp.mean((pred - target) ** 2)


@eqx.filter_jit
def train_step(model, opt_state, xy, target, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, xy, target)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def render_full(model: ImageNetwork, h: int, w: int) -> jnp.ndarray:
    coords = pixel_grid(h, w)
    pred = jax.vmap(model)(coords)
    return pred.reshape(h, w, 3)


def save_image(arr: jnp.ndarray, path: str) -> None:
    img = np.asarray(jnp.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    parser.add_argument("--downscale", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--out-dir", type=str, default="checkpoints/image")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log = setup_logger(args.out_dir)

    gt = load_image(args.image, downscale=args.downscale)
    h, w, _ = gt.shape
    log.info(f"loaded {args.image} at {w}x{h}")

    key = jax.random.PRNGKey(0)
    model_key, data_key = jax.random.split(key)
    model = ImageNetwork(model_key, ImageNetworkConfig())
    optimizer = build_adam(lr=args.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    pbar = tqdm(range(args.steps), desc="train")
    for step in pbar:
        data_key, sub = jax.random.split(data_key)
        xy = jax.random.uniform(sub, (args.batch_size, 2))
        target = jax.vmap(bilinear_sample, in_axes=(None, 0))(gt, xy)
        model, opt_state, loss = train_step(model, opt_state, xy, target, optimizer)
        pbar.set_postfix(loss=f"{float(loss):.6f}")

        if step % args.eval_interval == 0 or step == args.steps - 1:
            recon = render_full(model, h, w)
            full_mse = jnp.mean((recon - gt) ** 2)
            log.info(f"step {step:5d}  loss {loss:.6f}  psnr {float(psnr(full_mse)):.2f}dB")
            save_image(recon, os.path.join(args.out_dir, f"step_{step:05d}.png"))

    checkpoint.save(os.path.join(args.out_dir, "model.eqx"), model)
    log.info(f"saved checkpoint to {args.out_dir}/model.eqx")


if __name__ == "__main__":
    main()
