import torch
from torch import nn
from torch_model.message import Message
from torch_model.update_fn import Update
from torch_scatter import scatter

class PaiNN(nn.Module):

    def __init__(self, vocab_dim, feature_dim):
        super().__init__()
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
        
        return scatter(s, idx_m, dim=0).sum(dim=-1).squeeze()

if __name__ == "__main__":
    from experiment.config import BuilderConfigExperiment
    from transformers import set_seed
    from data.dataloader import construct_dataloaders
    set_seed(0)

    config = (
        BuilderConfigExperiment()
        .set_vocab_dim(20)
        .set_feature_dim(128)
        .set_train_size(100_000)
        .set_validation_size(15_000)
        .set_batch_size(2)
        .set_num_workers(6)
        .set_radius(3.)
        .set_path("./data/qm9.db")
        .set_split_file("./data/split2.npz")
        .build()
    )

    train, test, val = construct_dataloaders(config.train)

    data_point = (next(iter(train)))
    positions = data_point["_positions"]
    atomic_numbers = data_point["_atomic_numbers"]
    idx_i, idx_j = data_point["_idx_i"], data_point["_idx_j"]
    idx_m = data_point["_idx_m"]
    s0 = atomic_numbers[:, None]
    v0 = torch.zeros((len(atomic_numbers), 3, config.model.feature_dim))
    

    model = PaiNN(config.model.vocab_dim, config.model.feature_dim)

    dir_i_to_j = lambda x, i, j: x[j] - x[i]
    directions = dir_i_to_j(positions, idx_i, idx_j)

    ### Basic test ###

    predictions = model(v0, s0, directions, idx_i=idx_i, idx_j = idx_j, idx_m=idx_m)

    assert predictions.shape == (max(idx_m)+1,), (predictions, predictions.shape, max(idx_m) + 1) # s0.shape[:-1]
    print(predictions)
