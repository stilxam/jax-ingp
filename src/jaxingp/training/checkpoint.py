import equinox as eqx


def save(path: str, model: eqx.Module) -> None:
    eqx.tree_serialise_leaves(path, model)


def load(path: str, like: eqx.Module) -> eqx.Module:
    return eqx.tree_deserialise_leaves(path, like)
