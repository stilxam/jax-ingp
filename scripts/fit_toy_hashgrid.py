"""Fit MultiresHashGridEncoding(dim=2) -> Linear to a synthetic 2D pattern.

Isolates hash-grid gather/interpolation/backward correctness before any
MLP/NeRF complexity is layered on top. Loss should drop to near-zero within
a few hundred Adam steps if the encoding is wired correctly.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from PIL import Image

from jaxingp.encoding.hashgrid import MultiresHashGridEncoding


def target_fn(xy: jnp.ndarray) -> jnp.ndarray:
    x, y = xy[..., 0], xy[..., 1]
    r = jnp.stack([jnp.sin(8 * jnp.pi * x), jnp.cos(8 * jnp.pi * y), x * y], axis=-1)
    return 0.5 * (r + 1.0)


class ToyModel(eqx.Module):
    encoding: MultiresHashGridEncoding
    linear: eqx.nn.Linear

    def __init__(self, key):
        enc_key, lin_key = jax.random.split(key)
        self.encoding = MultiresHashGridEncoding(enc_key, dim=2)
        n_in = self.encoding.n_levels * self.encoding.n_features_per_level
        self.linear = eqx.nn.Linear(n_in, 3, key=lin_key)

    def __call__(self, xy):
        return jax.nn.sigmoid(self.linear(self.encoding(xy)))


def loss_fn(model, xy, target):
    pred = jax.vmap(model)(xy)
    return jnp.mean((pred - target) ** 2)


@eqx.filter_jit
def step(model, opt_state, xy, target, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, xy, target)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def main():
    key = jax.random.PRNGKey(0)
    model_key, data_key = jax.random.split(key)
    model = ToyModel(model_key)
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    n_steps = 500
    batch_size = 4096
    for step_idx in range(n_steps):
        data_key, sub = jax.random.split(data_key)
        xy = jax.random.uniform(sub, (batch_size, 2))
        target = target_fn(xy)
        model, opt_state, loss = step(model, opt_state, xy, target, optimizer)
        if step_idx % 50 == 0 or step_idx == n_steps - 1:
            print(f"step {step_idx:4d}  loss {loss:.6f}")

    res = 128
    grid = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0, 1, res), jnp.linspace(0, 1, res), indexing="ij"
        ),
        axis=-1,
    ).reshape(-1, 2)
    pred = jax.vmap(model)(grid).reshape(res, res, 3)
    target = target_fn(grid).reshape(res, res, 3)

    out = jnp.concatenate([target, pred], axis=1)
    out_img = Image.fromarray((jnp.clip(out, 0, 1) * 255).astype("uint8").__array__())
    out_path = "/tmp/fit_toy_hashgrid.png"
    out_img.save(out_path)
    print(f"saved target|prediction comparison to {out_path}")


if __name__ == "__main__":
    main()
