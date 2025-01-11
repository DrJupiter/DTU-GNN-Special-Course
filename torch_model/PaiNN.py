import torch
from torch import nn
from torch_model.message import Message
from torch_model.update_fn import Update
from torch_scatter import scatter_add

class PaiNN(nn.Module):

    def __init__(self, vocab_dim, feature_dim):
        self.embeddings = nn.Embedding(vocab_dim, feature_dim)

        self.lin_1 = nn.Linear(feature_dim, feature_dim)
        self.lin_2 = nn.Linear(feature_dim, 1)

        self.message_1 = Message(feature_dim, n=20, cutoff_radius=3.)
        self.message_2 = Message(feature_dim, n=20, cutoff_radius=3.)
        self.message_3 = Message(feature_dim, n=20, cutoff_radius=3.)
        
        self.update_1 = Update(feature_dim)
        self.update_2 = Update(feature_dim)
        self.update_3 = Update(feature_dim)

    def forward(self, v, s, r, idx_i, idx_j, idx_m):
        """
        v (Vector feautures): A x 3 x F
        s (Scalar feautures): A x 1 x F
        r (Directions): EDGES x 3
        """


        ### Embedding
        s = self.embeddings(s)

        #### 1 - messsage + update
        d_v, d_s = self.message_1(v, s, r, idx_i=idx_i, idx_j=idx_j)
        v, s = v + d_v, s + d_s
        
        d_v, d_s = self.update_1(v, s)
        v, s = v + d_v, s + d_s

        ### 2 - messsage + update
        d_v, d_s = self.message_1(v, s, r, idx_i=idx_i, idx_j=idx_j)
        v, s = v + d_v, s + d_s

        d_v, d_s = self.update_1(v, s)
        v, s = v + d_v, s + d_s

        #### 3 - messsage + update
        d_v, d_s = self.message_1(v, s, r, idx_i=idx_i, idx_j=idx_j)
        v, s = v + d_v, s + d_s

        d_v, d_s = self.update_1(v, s)
        v, s = v + d_v, s + d_s

        ### Linear layers
        s = self.lin_1(s)
        s = torch.nn.functional.silu(s)
        s = self.lin_2(s)
        # M x R x F
        
        return scatter_add(s, idx_m).sum(dim=-1).squeeze()

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
