import equinox as eqx
import jax
import jax.nn as jnn
from jaxtyping import Array, Float, PRNGKeyArray

from jaxingp.config import SdfNetworkConfig
from jaxingp.encoding.hashgrid import MultiresHashGridEncoding


class SdfNetwork(eqx.Module):
    """Hash-encoded position -> MLP -> scalar signed distance. Faithful port
    of instant-ngp's SDF primitive (testbed_sdf.cu): sphere tracing to a
    zero-crossing, not a large departure from CUDA like NeRF's marcher was."""

    encoding: MultiresHashGridEncoding
    mlp: eqx.nn.MLP

    def __init__(self, key: PRNGKeyArray, cfg: SdfNetworkConfig = SdfNetworkConfig()):
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
            out_size=1,
            width_size=cfg.width,
            depth=cfg.depth,
            activation=jnn.relu,
            key=mlp_key,
        )

    def __call__(self, x: Float[Array, "3"]) -> Float[Array, ""]:
        # x lives in [-1,1]^3 (MeshDataset's normalized mesh space); the hash
        # encoding indexes assume [0,1]^3, so remap before encoding.
        return self.mlp(self.encoding((x + 1.0) / 2.0))[0]
