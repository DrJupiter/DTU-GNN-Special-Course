import sys
from contextlib import contextmanager
@contextmanager
def temporary_sys_path(path):
    original_sys_path = sys.path.copy()
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path = original_sys_path

from model.linear_layer import construct_linear_layer
import jax

def construct_message(
        feature_dim: int, n=2, cutoff_radius=3, key=jax.random.PRNGKey(0), dtype=jax.numpy.float32
        ):

    key, *split = jax.random.split(key, 4)

    s_linear_layer_1, p_s_linear_layer_1 = construct_linear_layer(feature_dim,
                                                                  feature_dim,
                                                                  key=split[0],
                                                                  dtype=dtype
                                                                  )


    s_linear_layer_2, p_s_linear_layer_2 = construct_linear_layer(feature_dim,
                                                                  feature_dim * 3,
                                                                  key=split[1],
                                                                  dtype=dtype
                                                                  )

    r_linear_layer_1, p_r_linear_layer_1 = construct_linear_layer(n,
                                                                  feature_dim*3,
                                                                  key=split[2],
                                                                  dtype=dtype
                                                                  )
    radiual_range = jax.numpy.array(range(1, n+1))
    # B x n
    def radial_basis(norm_direction):
                                # n             *                          # B x 1 -> B x n /   # B x 1 -> B x n
        return jax.numpy.sin( radiual_range * jax.numpy.pi/cutoff_radius * norm_direction) / norm_direction

    def min_math(a, b):
        return (a + b - jax.numpy.abs(a - b))/2

    def f_cut(direction_embeddings):
        # B x F * 3 - 1 -> B x F * 3
        delta = direction_embeddings - cutoff_radius

        # B x F * 3 -> B x F * 3
        mask = min_math(delta, 0) * 1/delta

        # B x F * 3
        values = 0.5 * (
                jax.numpy.cos(jax.numpy.pi * direction_embeddings/cutoff_radius)
                + 1)

        # B x F * 3
        return values * mask


    parameters = {"s_linear_layer_1": p_s_linear_layer_1, "s_linear_layer_2": p_s_linear_layer_2,
                  "r_linear_layer_1": p_r_linear_layer_1}



    def message(parameters, vectors, scalars, directions):

        # phi
        # B x 1 x F -> B x 1 x F
        s = s_linear_layer_1(parameters["s_linear_layer_1"], scalars)
        s = jax.nn.silu(s)
        # B x 1 x F -> B x 1 x F * 3
        s = s_linear_layer_2(parameters["s_linear_layer_2"], s)

        # W

        # B x 3 -> B x 1
        norm_r = (jax.vmap(lambda x: (x @ x)**0.5)(directions)).reshape(-1, 1)

        # B x 1 -> B x n
        r = radial_basis(norm_r)

        # B x n -> B x F * 3
        r = r_linear_layer_1(parameters["r_linear_layer_1"], r)
        # question here is iff it should be f_cut(norm_r) * r?

        # B x F * 3 -> B x F * 3
        r = f_cut(r)
                #  B x 1 x F * 3 *  B x 1 x F * 3
        combined = s             *  r[:, jax.numpy.newaxis, :]

        # B x 1 x F, B x 1 x F, B x 1 x F
        delta_s, delta_v1, delta_v2 = jax.numpy.split(combined, 3, axis=-1)

                # B x 3 x F * B x 1 x F + # B x 1 x F * B x 3 x 1 -> B x 3 x F
        delta_v = vectors   * delta_v1 +    delta_v2  * (directions/norm_r)[:, :, jax.numpy.newaxis]

        # aggregate

        # question which axis should be summed accross?
        # B x 1 x F -> 1 x F
        delta_s = delta_s.sum(axis=0)
        # B x 3 x F -> 3 x F
        delta_v = delta_v.sum(axis=0)

                # 3 x F -> 1 x 3 x F               1 x F -> 1 x 1 x F
        return delta_v[jax.numpy.newaxis, :, :], delta_s[jax.numpy.newaxis, :, :]




    return message, parameters

if __name__ == "__main__":

    message, p_message = construct_message(10, n=20)

    scalars = jax.random.normal(jax.random.PRNGKey(0), (5, 1, 10))
    directions = jax.random.normal(jax.random.PRNGKey(0), (5, 3))
    vectors = jax.random.normal(jax.random.PRNGKey(0), (5, 3, 10))

    delta_s, delta_v = message(p_message, vectors, scalars, directions)

    message = jax.jit(message)

    message(p_message, vectors, scalars, directions)
