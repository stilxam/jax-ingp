import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from PIL import Image


def load_image(path: str, downscale: int = 1) -> Float[Array, "H W 3"]:
    img = Image.open(path).convert("RGB")
    if downscale > 1:
        img = img.resize((img.width // downscale, img.height // downscale), Image.LANCZOS)
    return jnp.asarray(np.asarray(img), dtype=jnp.float32) / 255.0


def bilinear_sample(image: Float[Array, "H W 3"], xy: Float[Array, "2"]) -> Float[Array, "3"]:
    """xy in [0, 1]^2, xy[0] = column fraction, xy[1] = row fraction."""
    h, w, _ = image.shape
    x = xy[0] * (w - 1)
    y = xy[1] * (h - 1)

    x0 = jnp.clip(jnp.floor(x).astype(jnp.int32), 0, w - 1)
    x1 = jnp.clip(x0 + 1, 0, w - 1)
    y0 = jnp.clip(jnp.floor(y).astype(jnp.int32), 0, h - 1)
    y1 = jnp.clip(y0 + 1, 0, h - 1)

    fx = x - x0
    fy = y - y0

    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bot = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def pixel_grid(h: int, w: int) -> Float[Array, "H*W 2"]:
    ys, xs = jnp.meshgrid(jnp.linspace(0, 1, h), jnp.linspace(0, 1, w), indexing="ij")
    return jnp.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1)
