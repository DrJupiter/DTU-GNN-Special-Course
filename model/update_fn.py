#%%
import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as nn

from model.linear_layer import construct_linear_layer

#%%

def construct_update_fn(feature_dim=128, key=jax.random.PRNGKey(42), dtype=jnp.float32):

    split_keys = jax.random.split(key,4)

    U_fn, U_weights   = construct_linear_layer(feature_dim,   feature_dim, bias=False, key=split_keys[0], dtype=dtype)
    V_fn, V_weights   = construct_linear_layer(feature_dim,   feature_dim, bias=False, key=split_keys[1], dtype=dtype)
    W1_fn, W1_weights = construct_linear_layer(2*feature_dim, feature_dim, bias=True,  key=split_keys[2], dtype=dtype)
    W2_fn, W2_weights = construct_linear_layer(feature_dim, 3*feature_dim, bias=True,  key=split_keys[3], dtype=dtype)

    weights = {
            "U_fn":U_weights,
            "V_fn": V_weights,
            "W1_fn": W1_weights,
            "W2_fn": W2_weights,
    }

    def update_fn(params, v, s):
        u = U_fn(params["U_fn"], v)
        v = V_fn(params["V_fn"], v)

        v_norm = jnp.linalg.norm(v, axis=-2, keepdims=True) # i do over axis=-1, he does axis=-2, but he has (N,1,feature_dim) shape. (makes no sense to norm over 1 dim)
        s = jnp.concatenate((s, v_norm), axis=-1)
        s = W1_fn(params["W1_fn"], s)
        s = nn.silu(s)
        s = W2_fn(params["W2_fn"], s)
        a_vv, a_sv, a_ss = jnp.split(s, 3, axis=-1)

        d_v = u*a_vv

        uv = jnp.sum(v * u, axis=1, keepdims=True) # according to eq. 9 (not fig. 2)
        d_s = uv*a_sv
        d_s = d_s+a_ss

        return d_v, d_s

    return update_fn, weights

#%%

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    split_keys = jax.random.split(key,4)

    N,feature_dim = 2,10

    update_fn, p_update_fn = construct_update_fn(feature_dim, split_keys[0])

    s = jax.random.normal(split_keys[1], (N, 1, feature_dim))
    v = jax.random.normal(split_keys[2], (N, 3, feature_dim))

    pred_v, pred_s = update_fn(p_update_fn, v, s)

    assert pred_v.shape == (N, 3, feature_dim) and pred_s.shape == (N, 1, feature_dim)

    update_fn = jax.jit(update_fn)

    pred_v, pred_s = update_fn(p_update_fn, v, s)

    assert pred_v.shape == (N, 3, feature_dim) and pred_s.shape == (N, 1, feature_dim)

    summ = lambda x,y: x.sum()+y.sum()

    loss_fn = lambda params, data, target: (
        (summ(*update_fn(params, data[0], data[1])) - target) ** 2
    ).mean()

    target = jax.random.normal(jax.random.PRNGKey(0), (1))

    data = [v,s]

    grad_fn = jax.grad(loss_fn)

    grads = grad_fn(p_update_fn, data, target)

    assert jnp.abs(1-jnp.isclose(jnp.array([jnp.abs(leaf).sum() for leaf in jax.tree.leaves(grads)]), 0)).prod()
    LEARNING_RATE = 1e-2

    p_update_fn = jax.tree.map(
        lambda p, g: p - LEARNING_RATE * g, p_update_fn, grads
    )
