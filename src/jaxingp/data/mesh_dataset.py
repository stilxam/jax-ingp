import jax
import jax.numpy as jnp
import numpy as np
import trimesh
from jaxtyping import Array, Float, PRNGKeyArray


class MeshDataset:
    """OBJ/STL mesh + on-the-fly signed-distance ground truth via trimesh's
    BVH (`trimesh.proximity.signed_distance`). A from-scratch performant BVH
    in JAX is a substantial side-project of its own, orthogonal to the goal
    of reimplementing instant-ngp's learned representations — trimesh is a
    pragmatic dependency for ground-truth generation only (CPU-side, off the
    hot training loop's critical path once batches are drawn)."""

    mesh: trimesh.Trimesh
    center: Float[Array, "3"]
    scale: float

    def __init__(self, mesh: trimesh.Trimesh, center: Float[Array, "3"], scale: float):
        self.mesh = mesh
        self.center = center
        self.scale = scale

    @staticmethod
    def load(path: str, margin: float = 1.1) -> "MeshDataset":
        mesh = trimesh.load(path, force="mesh")
        bounds = mesh.bounds  # (2,3)
        center = (bounds[0] + bounds[1]) / 2.0
        radius = np.max(bounds[1] - bounds[0]) / 2.0 * margin
        scale = 1.0 / radius  # normalizes mesh into roughly [-1,1]^3
        return MeshDataset(mesh, jnp.array(center, dtype=jnp.float32), float(scale))

    def to_mesh_space(self, pos: Float[Array, "... 3"]) -> Float[Array, "... 3"]:
        return pos / self.scale + self.center

    def sample_batch(
        self, key: PRNGKeyArray, batch_size: int, surface_frac: float = 0.5, surface_std: float = 0.01
    ) -> tuple[Float[Array, "batch_size 3"], Float[Array, "batch_size"]]:
        """Mix of uniform-volume and near-surface points (matches
        `generate_training_samples_sdf`), ground-truth signed distance
        computed via trimesh's BVH — a CPU-side (non-jitted) step."""
        n_surface = int(batch_size * surface_frac)
        n_uniform = batch_size - n_surface

        uniform_key, noise_key = jax.random.split(key, 2)
        uniform_pts = jax.random.uniform(uniform_key, (n_uniform, 3), minval=-1.0, maxval=1.0)

        surface_pts_mesh, _ = trimesh.sample.sample_surface(self.mesh, n_surface)
        surface_pts = jnp.asarray(np.asarray(surface_pts_mesh), dtype=jnp.float32)
        surface_pts = (surface_pts - self.center) * self.scale
        surface_pts = surface_pts + jax.random.normal(noise_key, surface_pts.shape) * surface_std

        points = jnp.concatenate([uniform_pts, surface_pts], axis=0)
        points_mesh_space = np.asarray(self.to_mesh_space(points))
        sdf_mesh_space = trimesh.proximity.signed_distance(self.mesh, points_mesh_space)
        # trimesh convention: positive = inside. instant-ngp/typical SDF convention:
        # negative = inside, positive = outside. Flip, and rescale distances into
        # our normalized [-1,1]-ish coordinate space.
        sdf = jnp.asarray(-sdf_mesh_space, dtype=jnp.float32) * self.scale

        return points, sdf
