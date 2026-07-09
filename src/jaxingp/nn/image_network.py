import equinox as eqx
import jax
import jax.nn as jnn
from jaxtyping import Array, Float, PRNGKeyArray

from jaxingp.config import ImageNetworkConfig
from jaxingp.encoding.hashgrid import MultiresHashGridEncoding


class ImageNetwork(eqx.Module):
    """2D gigapixel/image-fitting primitive: hash-encode (x, y) -> MLP -> RGB."""

    encoding: MultiresHashGridEncoding
    mlp: eqx.nn.MLP

    def __init__(self, key: PRNGKeyArray, cfg: ImageNetworkConfig = ImageNetworkConfig()):
        enc_key, mlp_key = jax.random.split(key)
        self.encoding = MultiresHashGridEncoding(
            enc_key,
            dim=cfg.pos_encoding.dim,
            n_levels=cfg.pos_encoding.n_levels,
            n_features_per_level=cfg.pos_encoding.n_features_per_level,
            log2_hashmap_size=cfg.pos_encoding.log2_hashmap_size,
            base_resolution=cfg.pos_encoding.base_resolution,
            per_level_scale=cfg.pos_encoding.per_level_scale,
            use_direct_indexing=cfg.pos_encoding.use_direct_indexing,
        )
        n_in = cfg.pos_encoding.n_levels * cfg.pos_encoding.n_features_per_level
        self.mlp = eqx.nn.MLP(
            in_size=n_in,
            out_size=3,
            width_size=cfg.width,
            depth=cfg.depth,
            activation=jnn.relu,
            key=mlp_key,
        )

    def __call__(self, xy: Float[Array, "2"]) -> Float[Array, "3"]:
        return jnn.sigmoid(self.mlp(self.encoding(xy)))
