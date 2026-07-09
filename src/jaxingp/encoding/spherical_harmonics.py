import jax.numpy as jnp
from jaxtyping import Array, Float

# Real spherical harmonics basis, l=0..3 (16 coefficients), standard closed-form
# polynomial basis (public math; matches the SphericalHarmonics(degree=4)
# encoding used throughout instant-ngp/tiny-cuda-nn, independent of source).
_C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def spherical_harmonics(d: Float[Array, "3"], degree: int = 4) -> Float[Array, "16"]:
    """Real SH basis of a unit direction vector, degree=4 -> l=0..3, 16 terms."""
    x, y, z = d[0], d[1], d[2]
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    terms = [_C0]
    if degree > 1:
        terms += [-_C1 * y, _C1 * z, -_C1 * x]
    if degree > 2:
        terms += [
            _C2[0] * xy,
            _C2[1] * yz,
            _C2[2] * (2.0 * zz - xx - yy),
            _C2[3] * xz,
            _C2[4] * (xx - yy),
        ]
    if degree > 3:
        terms += [
            _C3[0] * y * (3 * xx - yy),
            _C3[1] * xy * z,
            _C3[2] * y * (4 * zz - xx - yy),
            _C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
            _C3[4] * x * (4 * zz - xx - yy),
            _C3[5] * z * (xx - yy),
            _C3[6] * x * (xx - 3 * yy),
        ]
    return jnp.stack(terms)
