from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from jaxingp.geometry.aabb import BoundingBox


class SphereTraceResult(eqx.Module):
    hit_pos: Float[Array, "3"]
    hit: Bool[Array, ""]
    n_iter: Int[Array, ""]


class _Carry(eqx.Module):
    t: Float[Array, ""]
    hit: Bool[Array, ""]
    n_iter: Int[Array, ""]


def sphere_trace_ray(
    ray_o: Float[Array, "3"],
    ray_d: Float[Array, "3"],
    sdf_fn: Callable[[Float[Array, "3"]], Float[Array, ""]],
    aabb: BoundingBox,
    max_iters: int = 1000,
    eps: float = 1e-4,
) -> SphereTraceResult:
    """Classic sphere tracing: advance by the signed distance itself each
    step (safe since the surface can't be closer than |sdf(pos)|), until
    the distance is within `eps` of a zero-crossing or the ray exits the
    AABB / iteration budget is spent."""
    t0, t1 = aabb.ray_intersect(ray_o, ray_d)
    t0 = jnp.maximum(t0, 1e-4)

    init = _Carry(t=t0, hit=jnp.asarray(False), n_iter=jnp.zeros((), jnp.int32))

    def cond_fn(c: _Carry) -> Bool[Array, ""]:
        return (~c.hit) & (c.t < t1) & (c.n_iter < max_iters)

    def body_fn(c: _Carry) -> _Carry:
        pos = ray_o + c.t * ray_d
        d = sdf_fn(pos)
        hit = jnp.abs(d) < eps
        return _Carry(t=c.t + d, hit=hit, n_iter=c.n_iter + 1)

    final = jax.lax.while_loop(cond_fn, body_fn, init)
    hit_pos = ray_o + final.t * ray_d
    return SphereTraceResult(hit_pos, final.hit, final.n_iter)


def shade(
    sdf_fn: Callable[[Float[Array, "3"]], Float[Array, ""]],
    result: SphereTraceResult,
    light_dir: Float[Array, "3"],
    background: Float[Array, "3"],
    ambient: float = 0.15,
) -> Float[Array, "3"]:
    """Surface normal via autodiff (jax.grad of the network's own distance
    output) instead of CUDA's manual finite-difference/analytic gradient
    kernels — free in JAX. Simple Lambertian + ambient shading, sufficient
    for a visual correctness check (not the original's full path-traced/AO
    preview mode)."""
    normal = jax.grad(sdf_fn)(result.hit_pos)
    normal = normal / (jnp.linalg.norm(normal) + 1e-8)
    diffuse = jnp.maximum(jnp.dot(normal, light_dir), 0.0)
    shaded = jnp.full((3,), ambient + (1.0 - ambient) * diffuse)
    return jnp.where(result.hit, shaded, background)


def render_rays_sdf(
    sdf_fn: Callable[[Float[Array, "3"]], Float[Array, ""]],
    aabb: BoundingBox,
    rays_o: Float[Array, "n_rays 3"],
    rays_d: Float[Array, "n_rays 3"],
    light_dir: Float[Array, "3"],
    background: Float[Array, "3"],
    max_iters: int = 1000,
    eps: float = 1e-4,
) -> Float[Array, "n_rays 3"]:
    def render_one(ray_o, ray_d):
        result = sphere_trace_ray(ray_o, ray_d, sdf_fn, aabb, max_iters, eps)
        return shade(sdf_fn, result, light_dir, background)

    return jax.vmap(render_one)(rays_o, rays_d)
