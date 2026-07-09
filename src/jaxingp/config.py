from dataclasses import dataclass


@dataclass(frozen=True)
class HashGridConfig:
    dim: int = 3
    n_levels: int = 8
    n_features_per_level: int = 4
    log2_hashmap_size: int = 19
    base_resolution: int = 16
    per_level_scale: float = 2.0
    use_direct_indexing: bool = True


@dataclass(frozen=True)
class NerfNetworkConfig:
    pos_encoding: HashGridConfig = HashGridConfig(
        dim=3, n_levels=8, n_features_per_level=4, log2_hashmap_size=19, base_resolution=16
    )
    density_width: int = 64
    density_depth: int = 1
    rgb_width: int = 64
    rgb_depth: int = 2
    sh_degree: int = 4


@dataclass(frozen=True)
class SdfNetworkConfig:
    pos_encoding: HashGridConfig = HashGridConfig(
        dim=3, n_levels=16, n_features_per_level=2, log2_hashmap_size=19, base_resolution=16
    )
    width: int = 64
    depth: int = 2


@dataclass(frozen=True)
class ImageNetworkConfig:
    pos_encoding: HashGridConfig = HashGridConfig(
        dim=2, n_levels=8, n_features_per_level=4, log2_hashmap_size=19, base_resolution=16
    )
    width: int = 64
    depth: int = 2
