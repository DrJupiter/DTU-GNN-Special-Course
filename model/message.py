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
        feature_dim: int, key=jax.random.PRNGKey(0), dtype=jax.numpy.float32
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

    r_linear_layer_1, p_r_linear_layer_1 = construct_linear_layer(feature_dim,
                                                                  feature_dim*3,
                                                                  key=split[2],
                                                                  dtype=dtype
                                                                  )

    radial_basis = lambda x: x # TODO
    f_cut = lambda x: x # TODO

    parameters = {"s_linear_layer_1": p_s_linear_layer_1, "s_linear_layer_2": p_s_linear_layer_2,
                  "r_linear_layer_1": p_r_linear_layer_1}



    def message(parameters, scalars, directions):

        # phi
        s = s_linear_layer_1(parameters["s_linear_layer_1"], scalars)
        s = jax.nn.silu(s)
        s = s_linear_layer_2(parameters["s_linear_layer_2"], s)

        # W
        r = radial_basis(directions)
        r = r_linear_layer_1(parameters["p_r_linear_layer_1"], r)
        r = f_cut(r)

        combined = s * r
        delta_s, delta_v1, delta_v2 = jax.numpy.split(combined, 3, axis=-1)

        # aggregate



    return message, parameters

if __name__ == "__main__":

    message, p_message = construct_message(10)

    data = jax.random.normal(jax.random.PRNGKey(0), (3,10))

    message(p_message, data)

    message = jax.jit(message)

    message(p_message, data)
