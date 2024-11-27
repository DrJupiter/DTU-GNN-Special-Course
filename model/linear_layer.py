import jax


def construct_linear_layer(
    in_dim, out_dim, bias=True, key=jax.random.PRNGKey(0), dtype=jax.numpy.float32
):
    initializer = jax.nn.initializers.he_normal()
    key, *split = jax.random.split(key, 2 + bias)
    weight = initializer(split[0], (in_dim, out_dim), dtype=dtype)

    include_bias = bias

    if include_bias:
        bias = initializer(split[1], (1, out_dim), dtype=dtype)
        parameters = {"weight": weight, "bias": bias}

        def linear_layer(parameters, data):
            return data @ parameters["weight"] + parameters["bias"]
    else:
        parameters = {"weight": weight}

        def linear_layer(parameters, data):
            return data @ parameters["weight"] 

    return linear_layer, parameters


if __name__ == "__main__":
    linear_layer_1, p_linear_layer_1 = construct_linear_layer(5, 2)

    data = jax.random.normal(jax.random.PRNGKey(0), (3, 5))

    prediction = linear_layer_1(p_linear_layer_1, data)

    assert prediction.shape == (3, 2)

    linear_layer_1 = jax.jit(linear_layer_1)

    out = linear_layer_1(p_linear_layer_1, data)

    assert out.shape == (3, 2)

    loss_fn = lambda params, data, target: (
        (linear_layer_1(params, data) - target) ** 2
    ).mean()

    target = jax.random.normal(jax.random.PRNGKey(0), (3, 2))

    grad_fn = jax.grad(loss_fn)

    grads = grad_fn(p_linear_layer_1, data, target)

    LEARNING_RATE = 1e-2

    p_linear_layer_1 = jax.tree.map(
        lambda p, g: p - LEARNING_RATE * g, p_linear_layer_1, grads
    )
