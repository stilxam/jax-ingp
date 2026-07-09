import os

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from jaxingp.data.image_dataset import load_image
from jaxingp.data.transforms import load_transforms
from jaxingp.geometry.aabb import BoundingBox, aabb_from_scale, max_cascade_from_aabb_scale
from jaxingp.geometry.rays import camera_ray


class NerfDataset:
    images: Float[Array, "N H W 3"]
    c2w: Float[Array, "N 4 4"]
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    h: int
    w: int
    scale: float
    offset: Float[Array, "3"]
    aabb: BoundingBox
    max_cascade: int

    def __init__(self, images, c2w, fx, fy, cx, cy, k1, k2, p1, p2, h, w, scale, offset, aabb, max_cascade):
        self.images = images
        self.c2w = c2w
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.k1, self.k2, self.p1, self.p2 = k1, k2, p1, p2
        self.h, self.w = h, w
        self.scale = scale
        self.offset = offset
        self.aabb = aabb
        self.max_cascade = max_cascade

    @staticmethod
    def load(json_path: str, downscale: int = 1, n_cascades: int = 8) -> "NerfDataset":
        td = load_transforms(json_path)
        frames = [fr for fr in td.frames if os.path.exists(os.path.join(td.base_dir, fr.file_path))]
        images = jnp.stack(
            [load_image(os.path.join(td.base_dir, fr.file_path), downscale) for fr in frames]
        )
        c2w = jnp.stack([jnp.array(fr.transform_matrix) for fr in frames])

        h, w = images.shape[1], images.shape[2]
        s = 1.0 / downscale
        return NerfDataset(
            images=images,
            c2w=c2w,
            fx=td.fl_x * s,
            fy=td.fl_y * s,
            cx=td.cx * s,
            cy=td.cy * s,
            k1=td.k1,
            k2=td.k2,
            p1=td.p1,
            p2=td.p2,
            h=h,
            w=w,
            scale=td.scale,
            offset=jnp.array(td.offset),
            aabb=aabb_from_scale(td.aabb_scale, n_cascades),
            max_cascade=max_cascade_from_aabb_scale(td.aabb_scale),
        )

    def _ray_unit_space(self, img_idx, py, px):
        c2w = self.c2w[img_idx]
        o_world, d_world = camera_ray(
            c2w, px, py, self.fx, self.fy, self.cx, self.cy, self.k1, self.k2, self.p1, self.p2
        )
        o_unit = o_world * self.scale + self.offset
        d_unit = d_world * self.scale
        d_unit = d_unit / jnp.linalg.norm(d_unit)
        return o_unit, d_unit

    def sample_batch(self, key: PRNGKeyArray, batch_size: int, exclude_idx: int | None = None):
        n = self.images.shape[0]
        img_key, py_key, px_key = jax.random.split(key, 3)
        if exclude_idx is None:
            img_idx = jax.random.randint(img_key, (batch_size,), 0, n)
        else:
            img_idx = jax.random.randint(img_key, (batch_size,), 0, n - 1)
            img_idx = jnp.where(img_idx >= exclude_idx, img_idx + 1, img_idx)
        py = jax.random.uniform(py_key, (batch_size,), minval=0, maxval=self.h)
        px = jax.random.uniform(px_key, (batch_size,), minval=0, maxval=self.w)

        rays_o, rays_d = jax.vmap(self._ray_unit_space)(img_idx, py, px)

        py_i = jnp.clip(py.astype(jnp.int32), 0, self.h - 1)
        px_i = jnp.clip(px.astype(jnp.int32), 0, self.w - 1)
        target_rgb = self.images[img_idx, py_i, px_i]

        return rays_o, rays_d, target_rgb

    def render_rays_for_frame(self, img_idx: int):
        ys, xs = jnp.meshgrid(
            jnp.arange(self.h, dtype=jnp.float32),
            jnp.arange(self.w, dtype=jnp.float32),
            indexing="ij",
        )
        py = ys.reshape(-1)
        px = xs.reshape(-1)
        img_idx_arr = jnp.full_like(py, img_idx, dtype=jnp.int32)
        rays_o, rays_d = jax.vmap(self._ray_unit_space)(img_idx_arr, py, px)
        return rays_o, rays_d
