import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


def composite_ray(
    rgb: Float[Array, "n_samples 3"],
    density: Float[Array, "n_samples"],
    dts: Float[Array, "n_samples"],
    valid: Bool[Array, "n_samples"],
    background: Float[Array, "3"],
) -> tuple[Float[Array, "3"], Float[Array, ""]]:
    """Emission-absorption volume rendering. Padding slots (valid=False) get
    alpha=0 and contribute nothing to the sum, so no separate masking pass
    is needed downstream."""
    alpha = jnp.where(valid, 1.0 - jnp.exp(-density * dts), 0.0)
    trans = jnp.concatenate([jnp.ones((1,)), jnp.cumprod(1.0 - alpha + 1e-10)[:-1]])
    weight = alpha * trans
    acc = jnp.sum(weight)
    rgb_out = jnp.sum(weight[:, None] * rgb, axis=0) + (1.0 - acc) * background
    return rgb_out, acc
