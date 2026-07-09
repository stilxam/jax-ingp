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


def aabb_from_scale(aabb_scale: float, n_cascades: int) -> BoundingBox:
    """Training/render AABB from a dataset's `aabb_scale`, matching
    instant-ngp exactly (testbed_nerf.cu:2408-2436):
        m_aabb = BoundingBox{(0.5,0.5,0.5), (0.5,0.5,0.5)}
        m_aabb.inflate(0.5 * min(2**(n_cascades-1), aabb_scale))
    i.e. a box centered at (0.5,0.5,0.5) with side length
    `min(2**(n_cascades-1), aabb_scale)` — capped at the occupancy grid's
    total cascade range so the box never exceeds what the grid can index.
    """
    half_extent = 0.5 * min(2 ** (n_cascades - 1), aabb_scale)
    center = jnp.full(3, 0.5)
    return BoundingBox(center - half_extent, center + half_extent)


def max_cascade_from_aabb_scale(aabb_scale: float) -> int:
    """`ceil(log2(aabb_scale))`, matching instant-ngp's while-loop exactly
    (testbed_nerf.cu:2412-2414) — 0 when aabb_scale<=1. Clamps which of the
    occupancy grid's (always fully-allocated) cascades are actually used
    during marching; not traced, computed once at dataset-load time."""
    max_cascade = 0
    while (1 << max_cascade) < aabb_scale:
        max_cascade += 1
    return max_cascade
