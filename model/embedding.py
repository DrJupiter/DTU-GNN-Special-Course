import jax
import jax.numpy as jnp

from model.linear_layer import construct_linear_layer

def construct_embedding_fn(vocab_size=1, embed_dim=128, key=jax.random.PRNGKey(42), dtype=jnp.float32):
    """Simple lienar alyer"""

    split_keys = jax.random.split(key,2)

    W1_fn, W1_weights = construct_linear_layer(in_dim = vocab_size, out_dim = embed_dim, bias=False,  key=split_keys[0], dtype=dtype)

    weights = {
            "W1_fn": W1_weights,
    }

    def embed_fn(params, s):
        s = W1_fn(params["W1_fn"], s)
        return s

    return embed_fn, weights

if __name__ == "__main__" or True:
    
    ### init ###
    key = jax.random.PRNGKey(42)
    split_keys = jax.random.split(key,4)

    N, in_dim, feature_dim = 2, 10, 128
    s = jax.random.normal(split_keys[0], (N, in_dim))

    embed_fn, params = construct_embedding_fn(vocab_size=in_dim, embed_dim=feature_dim, key=split_keys[1])
    embedded_s = embed_fn(params, s)

    assert embedded_s.shape == (N,feature_dim)

    ### Jit test ###

    model = jax.jit(embed_fn)
    predictions = model(params, s)
    assert embedded_s.shape == (N,feature_dim)

    ### Grad test ###

    target = jax.random.normal(jax.random.PRNGKey(0), (N,feature_dim))

    loss_fn = lambda params, data, target: (
        (model(params, data) - target) ** 2
    ).mean()
    grad_fn = jax.grad(loss_fn)

    grads = grad_fn(params, s, target)
    LEARNING_RATE = 1e-2

    p_update_fn = jax.tree.map(
        lambda p, g: p - LEARNING_RATE * g, params, grads
    )

    assert jnp.abs(1-jnp.isclose(jnp.array([jnp.abs(leaf).sum() for leaf in jax.tree.leaves(grads)]), 0)).prod()