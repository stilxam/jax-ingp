"""Lightweight shape/dtype/NaN sanity checks for the hash grid encoding.
Not a pytest suite — a fast pre-flight script, run directly:
    JAX_PLATFORMS=cpu python tests/test_hashgrid_shapes.py
"""

import chex
import jax
import jax.numpy as jnp

from jaxingp.encoding.hashgrid import MultiresHashGridEncoding


def main():
    key = jax.random.PRNGKey(0)

    for dim in (2, 3):
        enc = MultiresHashGridEncoding(key, dim=dim, n_levels=8, n_features_per_level=4)
        x = jax.random.uniform(key, (16, dim))
        out = jax.vmap(enc)(x)
        chex.assert_shape(out, (16, 8 * 4))
        chex.assert_tree_all_finite(out)
        chex.assert_trees_all_equal_dtypes(out, jnp.zeros((), jnp.float32))

        grad = jax.grad(lambda m: jnp.sum(jax.vmap(m)(x) ** 2))(enc)
        chex.assert_trees_all_equal_shapes(grad.tables, enc.tables)
        assert bool(jnp.any(grad.tables != 0)), f"dim={dim}: expected nonzero gradient on tables"
        print(f"dim={dim}: shapes/dtype/finite/gradient ok")

    for use_direct in (True, False):
        enc = MultiresHashGridEncoding(key, dim=3, use_direct_indexing=use_direct)
        x = jax.random.uniform(key, (8, 3))
        out = jax.vmap(enc)(x)
        chex.assert_tree_all_finite(out)
        print(f"use_direct_indexing={use_direct}: finite ok")

    enc3 = MultiresHashGridEncoding(key, dim=3)
    assert any(enc3.is_direct), "expected some coarse levels to use direct indexing"
    assert not all(enc3.is_direct), "expected some fine levels to use hashing"
    print("is_direct mix (coarse=direct, fine=hashed): ok")

    print("all hashgrid checks passed")


if __name__ == "__main__":
    main()
