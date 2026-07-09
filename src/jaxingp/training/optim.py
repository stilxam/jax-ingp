import equinox as eqx
import jax
import jax.numpy as jnp
import optax


def build_adam(
    lr: float = 1e-2,
    weight_decay: float = 1e-6,
    b1: float = 0.9,
    b2: float = 0.99,
    eps: float = 1e-15,
) -> optax.GradientTransformation:
    return optax.chain(
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_adam(b1=b1, b2=b2, eps=eps),
        optax.scale(-lr),
    )


def build_nerf_schedule(
    lr: float = 1e-2,
    decay_start: int = 20000,
    decay_interval: int = 10000,
    decay_base: float = 0.33,
) -> optax.Schedule:
    def schedule(step):
        n_decays = jnp.floor(jnp.maximum(step - decay_start, 0) / decay_interval)
        return lr * decay_base**n_decays

    return schedule


def build_nerf_optimizer(
    lr: float = 1e-2,
    weight_decay: float = 1e-6,
    b1: float = 0.9,
    b2: float = 0.99,
    eps: float = 1e-15,
    decay_start: int = 20000,
    decay_interval: int = 10000,
    decay_base: float = 0.33,
) -> optax.GradientTransformation:
    schedule = build_nerf_schedule(lr, decay_start, decay_interval, decay_base)
    return optax.chain(
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_adam(b1=b1, b2=b2, eps=eps),
        optax.scale_by_schedule(lambda step: -schedule(step)),
    )


def ema_update(ema_model, model, decay: float):
    ema_params, static = eqx.partition(ema_model, eqx.is_array)
    params, _ = eqx.partition(model, eqx.is_array)
    new_params = jax.tree.map(lambda e, m: decay * e + (1 - decay) * m, ema_params, params)
    return eqx.combine(new_params, static)
