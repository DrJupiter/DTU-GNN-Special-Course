
import jax
import jax.numpy as jnp
import jax.nn as nn

# from model.linear_layer import construct_linear_layer
from model.linear_layer import construct_linear_layer
from model.update_fn import construct_update_fn
from model.message import construct_message
from model.embedding import construct_embedding_fn

def construct_PaiNN(vocab_dim=10, feature_dim=128, key=jax.random.PRNGKey(42), dtype=jnp.float32):

    split_keys = jax.random.split(key,1+2+3+3)

    embed_fn, embed_weights = construct_embedding_fn(vocab_dim, feature_dim, split_keys[0], dtype=dtype)

    W1_fn, W1_weights = construct_linear_layer(feature_dim, feature_dim, bias=True,  key=split_keys[1], dtype=dtype)
    W2_fn, W2_weights = construct_linear_layer(feature_dim, feature_dim, bias=True,  key=split_keys[2], dtype=dtype)

    M1_fn, M1_weights = construct_message(feature_dim,  key=split_keys[3], dtype=dtype)
    M2_fn, M2_weights = construct_message(feature_dim,  key=split_keys[4], dtype=dtype)
    M3_fn, M3_weights = construct_message(feature_dim,  key=split_keys[5], dtype=dtype)

    U1_fn, U1_weights = construct_update_fn(feature_dim, key=split_keys[6], dtype=dtype)
    U2_fn, U2_weights = construct_update_fn(feature_dim, key=split_keys[7], dtype=dtype)
    U3_fn, U3_weights = construct_update_fn(feature_dim, key=split_keys[8], dtype=dtype)

    weights = {
            "embed_fn":embed_weights,
            "W1_fn": W1_weights,
            "W2_fn": W2_weights,
            "M1_fn": M1_weights,
            "M2_fn": M2_weights,
            "M3_fn": M3_weights,
            "U1_fn": U1_weights,
            "U2_fn": U2_weights,
            "U3_fn": U3_weights,
    }

    def PaiNN(weights, v, s, r, idx_i, idx_j, idx_m):
        """

        v (Vector feautures): A x 3 x F
        s (Scalar feautures): A x 1 x F
        r (Directions): EDGES x 3

        """

        ### Embedding
        s = embed_fn(weights["embed_fn"], s)

        ### 1 - messsage + update
        d_v,d_s = M1_fn(weights["M1_fn"], v, s, r, idx_i=idx_i, idx_j=idx_j)
        v = d_v + v
        s = d_s + s

        d_v,d_s = U1_fn(weights["U1_fn"], v, s)
        v = d_v + v
        s = d_s + s

        ## 2 - messsage + update
        d_v,d_s = M2_fn(weights["M2_fn"], v, s, r, idx_i=idx_i, idx_j=idx_j)
        v = d_v + v
        s = d_s + s

        d_v,d_s = U2_fn(weights["U2_fn"], v, s)
        v = d_v + v
        s = d_s + s

        ### 3 - messsage + update
        d_v,d_s = M3_fn(weights["M3_fn"], v, s, r, idx_i=idx_i, idx_j=idx_j)
        v = d_v + v
        s = d_s + s

        d_v,d_s = U3_fn(weights["U3_fn"], v, s)
        v = d_v + v # TODO: asses if needed
        s = d_s + s

        ### Linear layers
        s = W1_fn(weights["W1_fn"],s)
        s = nn.silu(s)
        s = W2_fn(weights["W2_fn"],s)
        # M x R x F
        return jax.numpy.squeeze(jax.ops.segment_sum(s, idx_m).sum(axis=-1)) # Index sum, sum over each molecule # see jax.ops.segment_sum

    return PaiNN, weights

if __name__ == "__main__":
    import jax
    from experiment.config import BuilderConfigExperiment
    from transformers import set_seed
    from data.dataloader import construct_dataloaders
    set_seed(0)

    config = (
        BuilderConfigExperiment()
        .set_vocab_dim(20)
        .set_feature_dim(256)
        .set_train_size(150_000)
        .set_validation_size(15_000)
        .set_batch_size(2)
        .set_num_workers(6)
        .set_radius(3.)
        .set_path("./data/qm9.db")
        .set_split_file("./data/split.npz")
        .build()
    )

    train, test, val = construct_dataloaders(config.train)

    data_point = (next(iter(train)))
    positions = jax.numpy.array(data_point["_positions"].numpy())
    atomic_numbers = jax.numpy.array(data_point["_atomic_numbers"].numpy())
    idx_i, idx_j = jax.numpy.array(data_point["_idx_i"].numpy()), jax.numpy.array(data_point["_idx_j"].numpy())
    idx_m = jax.numpy.array(data_point["_idx_m"].numpy())
    s0 = atomic_numbers[:, jax.numpy.newaxis]
    v0 = jax.numpy.zeros((len(atomic_numbers), 3, config.model.feature_dim))


    ### init ###
    #N, feature_dim, vocab_dim = 3, 128, 10

    model, params = construct_PaiNN(vocab_dim=config.model.vocab_dim, feature_dim=config.model.feature_dim)

    key = jax.random.PRNGKey(42)
    split_keys = jax.random.split(key,4)

    #v0 = jax.random.normal(split_keys[0], (N, 3, feature_dim))
    #s0 = jax.random.normal(split_keys[1], (N, 1, vocab_dim))
    #positions = jax.random.normal(split_keys[2], (N, 3))

    #idx_i = jax.numpy.array([0, 0, 1, 2])
    #idx_j = jax.numpy.array([1,2,0,0])
    #idx_m = jax.numpy.array([0,0,1])

    dir_i_to_j = lambda x, i, j: x[j] - x[i]
    directions = dir_i_to_j(positions, idx_i, idx_j)

    ### Basic test ###

    predictions = model(params, v0, s0, directions, idx_i=idx_i, idx_j = idx_j, idx_m=idx_m)

    assert predictions.shape == (max(idx_m)+1,), (predictions, predictions.shape, max(idx_m) + 1) # s0.shape[:-1]

    ### Jit test ###

    #model = jax.jit(model)
    #predictions = model(params, v0, s0, r)
    #assert predictions.shape == (N,1)

    ### Grad test ###

    target = jax.random.normal(jax.random.PRNGKey(0), (max(idx_m) + 1,))
    target = jax.numpy.array(data_point['energy_U0'].numpy())
    data = [v0,s0,directions, idx_i, idx_j, idx_m]

    loss_fn = lambda params, data, target: (
        (model(params, *data) - target) ** 2
    ).mean()
    grad_fn = jax.grad(loss_fn)

    grads = grad_fn(params, data, target)
    LEARNING_RATE = 1e-2

    p_update_fn = jax.tree.map(
        lambda p, g: p - LEARNING_RATE * g, params, grads
    )

    # Check that all layers's weights are being updated
    assert jnp.abs(1-jnp.isclose(jnp.array([jnp.abs(leaf).sum() for leaf in jax.tree.leaves(grads)]), 0)).prod()
