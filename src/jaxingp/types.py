from jaxtyping import Array, Float, PRNGKeyArray

Vec2 = Float[Array, "2"]
Vec3 = Float[Array, "3"]
RGB = Float[Array, "3"]

__all__ = ["Array", "Float", "PRNGKeyArray", "Vec2", "Vec3", "RGB"]
