import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from jaxingp.config import NerfNetworkConfig
from jaxingp.encoding.hashgrid import MultiresHashGridEncoding
from jaxingp.encoding.spherical_harmonics import spherical_harmonics


class NerfNetwork(eqx.Module):
    """Two-MLP NeRF network: hash-encoded position -> density + 15-dim geo
    feature; SH-encoded direction + geo feature -> RGB."""

    pos_encoding: MultiresHashGridEncoding
    density_mlp: eqx.nn.MLP
    rgb_mlp: eqx.nn.MLP
    sh_degree: int = eqx.field(static=True)

    def __init__(self, key: PRNGKeyArray, cfg: NerfNetworkConfig = NerfNetworkConfig()):
        enc_key, density_key, rgb_key = jax.random.split(key, 3)
        self.pos_encoding = MultiresHashGridEncoding(
            enc_key,
            dim=cfg.pos_encoding.dim,
            n_levels=cfg.pos_encoding.n_levels,
            n_features_per_level=cfg.pos_encoding.n_features_per_level,
            log2_hashmap_size=cfg.pos_encoding.log2_hashmap_size,
            base_resolution=cfg.pos_encoding.base_resolution,
            per_level_scale=cfg.pos_encoding.per_level_scale,
            use_direct_indexing=cfg.pos_encoding.use_direct_indexing,
        )
        n_pos_feat = cfg.pos_encoding.n_levels * cfg.pos_encoding.n_features_per_level
        self.density_mlp = eqx.nn.MLP(
            in_size=n_pos_feat,
            out_size=16,
            width_size=cfg.density_width,
            depth=cfg.density_depth,
            activation=jnn.relu,
            key=density_key,
        )
        self.sh_degree = cfg.sh_degree
        n_dir_feat = cfg.sh_degree * cfg.sh_degree
        self.rgb_mlp = eqx.nn.MLP(
            in_size=n_dir_feat + 15,
            out_size=3,
            width_size=cfg.rgb_width,
            depth=cfg.rgb_depth,
            activation=jnn.relu,
            key=rgb_key,
        )

    def density_raw(self, x: Float[Array, "3"]) -> Float[Array, "16"]:
        return self.density_mlp(self.pos_encoding(x))

    def density(self, x: Float[Array, "3"]) -> Float[Array, ""]:
        raw = self.density_raw(x)[0]
        return jnp.exp(jnp.clip(raw, -15.0, 15.0))

    def __call__(
        self, x: Float[Array, "3"], d: Float[Array, "3"]
    ) -> tuple[Float[Array, "3"], Float[Array, ""]]:
        rgbd_feat = self.density_raw(x)
        density = jnp.exp(jnp.clip(rgbd_feat[0], -15.0, 15.0))
        geo_feat = rgbd_feat[1:]
        dir_enc = spherical_harmonics(d, degree=self.sh_degree)
        rgb_in = jnp.concatenate([dir_enc, geo_feat])
        rgb = jnn.sigmoid(self.rgb_mlp(rgb_in))
        return rgb, density
