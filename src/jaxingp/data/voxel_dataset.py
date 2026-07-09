import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


def load_voxel_grid(path: str) -> Float[Array, "D H W 4"]:
    """Loads a raw (D, H, W, 4) RGB+density array from .npy."""
    return jnp.asarray(np.load(path), dtype=jnp.float32)


def synthesize_toy_volume(resolution: int = 64) -> Float[Array, "D H W 4"]:
    """No real volume dataset ships in this checkout (see PLAN.md: original
    instant-ngp distills NanoVDB volumes, deliberately out of scope here).
    Synthesizes a colored sphere with a soft density falloff, sufficient to
    exercise the trilinear-sample + march + composite renderer."""
    coords = jnp.linspace(0.0, 1.0, resolution)
    zz, yy, xx = jnp.meshgrid(coords, coords, coords, indexing="ij")
    center = jnp.array([0.5, 0.5, 0.5])
    r = jnp.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2)

    radius = 0.35
    softness = 0.03
    density = 40.0 / (1.0 + jnp.exp((r - radius) / softness))

    rgb = jnp.stack(
        [0.3 + 0.7 * xx, 0.3 + 0.7 * yy, 0.3 + 0.7 * (1.0 - zz)], axis=-1
    )
    return jnp.concatenate([rgb, density[..., None]], axis=-1)
