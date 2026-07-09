# jaxingp

A from-scratch reimplementation of [NVIDIA instant-ngp](https://github.com/NVlabs/instant-ngp) (Instant Neural Graphics Primitives) in [JAX](https://github.com/jax-ml/jax), using [Equinox](https://github.com/patrick-kidger/equinox) for modules and [jaxtyping](https://github.com/patrick-kidger/jaxtyping) for array type annotations.

Four primitives, all built on the same multiresolution hash grid encoding:

- **NeRF** — 3D volumetric scene reconstruction from posed images, with occupancy-grid adaptive ray marching.
- **Image** — 2D gigapixel/image fitting (hash grid + MLP mapping (x,y) → RGB).
- **SDF** — signed distance field fit to a mesh, rendered via sphere tracing.
- **Volume** — direct ray-marched rendering of a precomputed density/color voxel grid (no training).

Grounded in a direct reading of the NVIDIA reference implementation (`tiny-cuda-nn`/`instant-ngp` CUDA source), not copied from it — the hash encoding, spherical harmonics, occupancy grid, and marchers are re-derived from the public method description and adapted to fit JAX's tracing model.

## Design notes & deliberate deviations from the CUDA original

- **Ray marching** (`render/march.py`): CUDA uses a two-pass count-then-compact scheme (`atomicAdd`-based `numsteps`/`ray_indices`) requiring data-dependent output shapes, which XLA's static-shape tracing can't express. This implementation instead runs `jax.vmap(jax.lax.while_loop)` per ray, with each ray writing into a fixed-size `max_samples`-length padded/masked output slot — functionally equivalent adaptive marching (each ray still runs its own data-dependent iteration count), but with zero global index bookkeeping.
- **Step size**: CUDA's closed-form log-space `to_stepping_space`/`advance_n_steps` exists to jump an arbitrary number of steps in O(1), which only matters for the compaction scheme above. Since the while_loop already advances one iteration at a time, step size is instead `cone_angle_min_stepsize * 2**cascade_mip` directly — same geometric growth envelope, simpler implementation. Voxel-boundary skipping (`_distance_to_next_voxel`, exact DDA) *is* ported faithfully, since that's what makes empty-space skipping efficient.
- **Occupancy grid** (`occupancy/grid.py`): a dense `(n_cascades, res, res, res)` array rather than a packed Morton-order bitfield — the Z-order layout exists in CUDA purely for warp memory coherence, which has no XLA analogue, and a dense float array is trivially affordable and simpler to gather from. Occupancy is an *optical-thickness* test (`density * dt_at_cascade > threshold`), not a raw density threshold — a fixed density cutoff, ignoring the actual step size taken, flags nearly everything as "occupied" once density is nonzero anywhere.
- **Grid bootstrapping** (`mark_untrained_density_grid`): a freshly-initialized network has near-uniform density, which the optical-thickness test correctly (but unhelpfully) reads as empty everywhere, so the marcher would find zero samples on every ray and the density MLP would never receive a gradient — a permanent dead state. Mirroring CUDA's `mark_untrained_density_grid`: cells inside any training camera's view frustum are seeded as guaranteed-occupied (large density, corrected downward by real evaluations later); cells no camera can ever see are seeded with a negative sentinel and *permanently* excluded from the EMA update (not just during warmup).
- **Scene scale**: instead of CUDA's fixed `NERF_SCALE=0.33` / `offset=(0.5,0.5,0.5)` convention (tuned for NeRF-synthetic-scale scenes), the world-to-unit-cube transform is auto-fit from the training camera positions (`geometry/aabb.py:fit_scene_transform`) — a uniform scale + offset (so directions/distances aren't skewed) centered on the camera cluster with a margin. Works regardless of the dataset's native scale.
- **Hash encoding** (`encoding/hashgrid.py`): a single dense `2**log2_hashmap_size`-sized table per level (even for coarse levels that need far fewer entries) — wastes some memory but keeps the table a clean, uniformly-shaped array. No custom CUDA-style atomic-add backward kernel: the corner-weighted gather is plain autodiff-differentiable, and XLA's scatter-add gradient rule handles hash-collision gradient accumulation for free.
- **Volume primitive** (`render/volume_march.py`): CUDA's volume mode distills a NanoVDB density grid into a HashGrid+MLP via Monte-Carlo delta-tracking/path-traced supervision — a large extra subsystem (NanoVDB binary parsing + MC path integrator) for a payoff (network compression of the volume) orthogonal to validating the hash-encoding+MLP+marching stack, which the NeRF/SDF/Image primitives already exercise. This implementation treats Volume as **direct-rendering, non-learned**: load a raw voxel array, ray-march with trilinear sampling, composite — no training, no NanoVDB.
- **Training loop efficiency**: `training/train_nerf.py` runs each `--eval-interval` block of steps as a single `jax.lax.scan`-compiled chunk (occupancy grid update folded in via `lax.cond`) rather than a Python loop calling a jitted step function once per step — one host↔device sync per eval interval instead of one per step, which matters once each step is fast relative to Python dispatch overhead.

## Setup

```sh
uv sync
```

Requires a CUDA-capable GPU for reasonable training speed (`jax[cuda13]`); everything also runs on CPU (`JAX_PLATFORMS=cpu`), just slower.

## Usage

### Hash grid encoding sanity check
```sh
uv run python scripts/fit_toy_hashgrid.py
```

### Image primitive
```sh
uv run python -m jaxingp.training.train_image path/to/image.jpg --downscale 4 --steps 2000
```

### NeRF
Expects a `transforms.json` dataset in the NeRF-synthetic / instant-ngp format: `fl_x`/`fl_y` or `camera_angle_x`, `cx`/`cy`, `w`/`h`, optional `k1,k2,p1,p2` lens distortion, `frames: [{file_path, transform_matrix}]` (4x4 camera-to-world). The [instant-ngp fox example dataset](https://github.com/NVlabs/instant-ngp/tree/master/data/nerf/fox) is a good starting point if you don't have your own capture.
```sh
uv run python -m jaxingp.training.train_nerf path/to/transforms.json \
    --marcher adaptive --downscale 8 --steps 3000
uv run python scripts/render_novel_view.py path/to/transforms.json checkpoints/nerf --frame-idx 0
```
`--marcher uniform` bypasses the occupancy grid (stratified sampling across the whole AABB per ray) — useful as a simpler baseline.

### Volume
No training — renders a synthesized toy volume (or `--voxel-path` to a `.npy` RGB+density array) from an orbit camera.
```sh
uv run python scripts/render_volume.py --out /tmp/render_volume.png
```

### SDF
Any watertight OBJ/STL mesh works (e.g. the [Stanford bunny](https://graphics.stanford.edu/data/3Dscanrep/) or `trimesh.creation` primitives).
```sh
uv run python -m jaxingp.training.train_sdf path/to/mesh.obj --steps 2000
uv run python scripts/render_sdf.py checkpoints/sdf/model.eqx --out /tmp/render_sdf.png
```

## Tests

Lightweight shape/dtype/NaN sanity scripts (not a pytest suite):
```sh
JAX_PLATFORMS=cpu uv run python tests/test_hashgrid_shapes.py
JAX_PLATFORMS=cpu uv run python tests/test_march_shapes.py
```

## Layout

```
src/jaxingp/
  encoding/    hash grid + spherical harmonics encodings
  nn/          NeRF / Image / SDF network modules
  geometry/    AABB, camera ray generation, orbit camera helpers
  occupancy/   cascaded occupancy grid (NeRF)
  render/      ray marching + volume rendering for each primitive
  data/        dataset loaders (transforms.json, images, meshes, voxel grids)
  training/    training loops, optimizer/EMA setup, checkpointing
scripts/       standalone rendering/validation scripts
tests/         shape sanity checks
```

## License

[NVIDIA Source Code License](./LICENSE.txt) (non-commercial use only) — adopted to match instant-ngp's own license, since this project borrows specific hyperparameters and design decisions from reading its source, even though no code was copied.
