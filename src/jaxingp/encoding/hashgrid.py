import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, UInt32

# Canonical instant-ngp spatial-hash primes (first `dim` are used).
_PRIMES = (1, 2654435761, 805459861, 3674653429)


class MultiresHashGridEncoding(eqx.Module):
    """Multiresolution hash grid encoding (instant-ngp), dim-parameterized (2D or 3D).

    Feature tables are trainable; per-level grid resolutions, hash-vs-direct
    indexing, and corner offsets are static (precomputed once, independent of
    dim being 2 or 3).
    """

    tables: Float[Array, "n_levels table_size n_features"]

    dim: int = eqx.field(static=True)
    n_levels: int = eqx.field(static=True)
    n_features_per_level: int = eqx.field(static=True)
    log2_hashmap_size: int = eqx.field(static=True)
    resolutions: tuple = eqx.field(static=True)
    is_direct: tuple = eqx.field(static=True)
    corner_offsets: tuple = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        dim: int,
        n_levels: int = 8,
        n_features_per_level: int = 4,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        per_level_scale: float = 2.0,
        use_direct_indexing: bool = True,
    ):
        self.dim = dim
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size

        table_size = 2**log2_hashmap_size
        resolutions = tuple(
            int(base_resolution * per_level_scale**level) for level in range(n_levels)
        )
        self.resolutions = resolutions
        self.is_direct = tuple(
            use_direct_indexing and (res + 1) ** dim <= table_size for res in resolutions
        )
        self.corner_offsets = tuple(itertools.product((0, 1), repeat=dim))

        self.tables = jax.random.uniform(
            key, (n_levels, table_size, n_features_per_level), minval=-1e-4, maxval=1e-4
        )

    def __call__(
        self, x: Float[Array, " dim"]
    ) -> Float[Array, " n_levels*n_features_per_level"]:
        return jnp.concatenate(
            [self._encode_level(x, level) for level in range(self.n_levels)], axis=-1
        )

    def _direct_index(self, coord: Int[Array, " dim"], resolution: int) -> UInt32[Array, ""]:
        idx = jnp.zeros((), jnp.uint32)
        stride = 1
        for d in range(self.dim):
            idx = idx + jnp.clip(coord[d], 0, resolution).astype(jnp.uint32) * jnp.uint32(stride)
            stride *= resolution + 1
        return idx

    def _spatial_hash(self, coord: Int[Array, " dim"]) -> UInt32[Array, ""]:
        c = coord.astype(jnp.uint32)
        h = jnp.zeros((), jnp.uint32)
        for d in range(self.dim):
            h = h ^ (c[d] * jnp.uint32(_PRIMES[d]))
        return h % jnp.uint32(self.tables.shape[1])

    def _encode_level(self, x: Float[Array, " dim"], level: int) -> Float[Array, " n_features"]:
        resolution = self.resolutions[level]
        x_scaled = x * resolution
        pos0 = jnp.floor(x_scaled).astype(jnp.int32)
        frac = x_scaled - pos0

        offsets = jnp.array(self.corner_offsets, dtype=jnp.int32)  # (n_corners, dim)
        corners = jnp.clip(pos0[None, :] + offsets, 0, resolution)  # (n_corners, dim)

        if self.is_direct[level]:
            indices = jax.vmap(lambda c: self._direct_index(c, resolution))(corners)
        else:
            indices = jax.vmap(self._spatial_hash)(corners)
        feats = self.tables[level][indices]  # (n_corners, n_features)

        weights = jnp.prod(
            jnp.where(offsets == 1, frac[None, :], 1.0 - frac[None, :]), axis=-1
        )  # (n_corners,)
        return jnp.sum(weights[:, None] * feats, axis=0)
