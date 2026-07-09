import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class BoundingBox(eqx.Module):
    """Axis-aligned box in the network's warped [0,1]^3 coordinate space."""

    min_corner: Float[Array, "3"]
    max_corner: Float[Array, "3"]

    def __init__(
        self,
        min_corner: Float[Array, "3"] = jnp.zeros(3),
        max_corner: Float[Array, "3"] = jnp.ones(3),
    ):
        self.min_corner = min_corner
        self.max_corner = max_corner

    def ray_intersect(
        self, o: Float[Array, "3"], d: Float[Array, "3"]
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        inv_d = 1.0 / jnp.where(d == 0, 1e-12, d)
        t_lo = (self.min_corner - o) * inv_d
        t_hi = (self.max_corner - o) * inv_d
        t_near = jnp.minimum(t_lo, t_hi)
        t_far = jnp.maximum(t_lo, t_hi)
        t0 = jnp.max(t_near)
        t1 = jnp.min(t_far)
        return t0, t1

    def contains(self, pos: Float[Array, "3"]) -> Float[Array, ""]:
        return jnp.all((pos >= self.min_corner) & (pos <= self.max_corner))


def fit_scene_transform(
    camera_positions: Float[Array, "N 3"], margin: float = 1.25
) -> tuple[Float[Array, ""], Float[Array, "3"]]:
    """Auto-fit a uniform scale + offset mapping camera positions into [0,1]^3.

    Deliberate simplification vs. instant-ngp's fixed NERF_SCALE=0.33 /
    offset=(0.5,0.5,0.5) convention (which assumes normalized NeRF-synthetic
    scene units): instead fit a similarity transform (uniform scale, so
    distances/directions aren't skewed per-axis) from the actual camera
    positions, centered at 0.5 with a margin so the scene fits comfortably
    inside the unit cube regardless of the dataset's native scale.
    """
    center = jnp.mean(camera_positions, axis=0)
    radius = jnp.max(jnp.linalg.norm(camera_positions - center, axis=-1)) * margin
    scale = 0.5 / radius
    offset = 0.5 - center * scale
    return scale, offset
