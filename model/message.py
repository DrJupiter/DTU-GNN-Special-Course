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
        feature_dim: int, n=20, cutoff_radius=3, key=jax.random.PRNGKey(0), dtype=jax.numpy.float32
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

    # EDGES x 1 -> EDGES x n
    def radial_basis(norm_direction):
                                # n             *                          # EDGES x 1 -> EDGES x n /   # EDGES x 1 -> EDGES x n
        return jax.numpy.sin( radiual_range * jax.numpy.pi/cutoff_radius * norm_direction) / norm_direction

    def min_math(a, b):
        return (a + b - jax.numpy.abs(a - b))/2

    # define mask on r_ij and apply it after linear layer,
    #   to only focus on the values that are relevant
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



    def message(parameters, vectors, scalars, directions, idx_i, idx_j):

        """
        vectors: A x 3 x F
        scalars: A x 1 x F
        directions: EDGES x 3
        """

        # phi
        # A x 1 x F -> A x 1 x F
        s = s_linear_layer_1(parameters["s_linear_layer_1"], scalars)
        s = jax.nn.silu(s)
        # A x 1 x F -> A x 1 x F * 3
        s = s_linear_layer_2(parameters["s_linear_layer_2"], s)

        # W

        # EDGES x 3 -> EDGES x 1
        norm_r = (jax.vmap(lambda x: (x @ x)**0.5)(directions)).reshape(-1, 1)

        # A -> B = B - A

        # EDGES x 1 -> EDGES x n
        r = radial_basis(norm_r)

        # EDGES x n -> EDGES x F * 3
        r = r_linear_layer_1(parameters["r_linear_layer_1"], r)

        # EDGES x F * 3 -> EDGES x F * 3
        r = f_cut(norm_r) * r

                #  EDGES x 1 x F * 3    *  EDGES x 1 x F * 3
        combined = s[idx_i]             *  r[:, jax.numpy.newaxis, :]

        # EDGES x 1 x F, EDGES x 1 x F, EDGES x 1 x F
        delta_s, delta_v1, delta_v2 = jax.numpy.split(combined, 3, axis=-1)

                # A x 3 x F        * EDGES x 1 x F +    EDGES x 1 x F *  EDGES x 3 x 1 -> EDGES x 3 x F
        delta_v = vectors[idx_i]   * delta_v1      +    delta_v2      * (directions/norm_r)[:, :, jax.numpy.newaxis]
        # EDGES x 3 x F -> A x 3 x F

        # aggregate

        # EDGES x 1 x F -> A x 1 x F
        delta_s = edge_segment_sum(idx_j, delta_s)
        # EDGES x 3 x F -> A x 3 x F
        delta_v = edge_segment_sum(idx_j, delta_v)

        assert delta_v.shape == vectors.shape
        assert delta_s.shape == delta_s.shape

        return delta_v, delta_s

    return message, parameters

                                #  EDGES x E x R
def edge_segment_sum(idx_j, features):
    """
    We sum over j, because we go from i -> j in building our directions
    and atom features.
    """
                                 # Edges x E x R
    result = jax.ops.segment_sum(features, idx_j)
                          # number of atoms, we assume every atom has at least one connection.
    assert len(result) == jax.numpy.max(idx_j) + 1
    return result

if __name__ == "__main__":

    message, p_message = construct_message(10, n=20)

    scalars = jax.random.normal(jax.random.PRNGKey(0), (3, 1, 10))
    positions = jax.random.normal(jax.random.PRNGKey(0), (3, 3))
    vectors = jax.random.normal(jax.random.PRNGKey(0), (3, 3, 10))
    idx_i = jax.numpy.array([0, 0, 1, 2])
    idx_j = jax.numpy.array([1,2,0,0])

    dir_i_to_j = lambda x, i, j: x[j] - x[i]
    directions = dir_i_to_j(positions, idx_i, idx_j)

    delta_s, delta_v = message(p_message, vectors, scalars, directions, idx_i, idx_j)

    #message = jax.jit(message)

    #message(p_message, vectors, scalars, directions, idx_i, idx_j)
