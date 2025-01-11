import torch
from torch import nn

class Update(nn.Module):

    def __init__(self, feature_dim):
        super().__init__()

        self.lin_u = nn.Linear(feature_dim, feature_dim, bias=False)
        self.lin_v = nn.Linear(feature_dim, feature_dim, bias=False)

        self.lin_1 = nn.Linear(2 * feature_dim, feature_dim)
        self.lin_2 = nn.Linear(feature_dim, 3 * feature_dim)
        self.feature_dim = feature_dim

    def forward(self, vectors, scalars):
        u = self.lin_u(vectors)
        v = self.lin_v(vectors)

        v_norm = torch.linalg.norm(v, dim=-2, keepdim=True)

        s = torch.concat((scalars, v_norm), dim=-1)
        s = self.lin_1(s)
        s = torch.nn.functional.silu(s)
        s = self.lin_2(s)
        a_vv, a_sv, a_ss = torch.split(s, self.feature_dim, dim=-1)

        d_v = u * a_vv
        uv = torch.sum(v * u, dim=1, keepdim=True)

        d_s = uv * a_sv
        d_s+= a_ss

        return d_v, d_s


if __name__ == "__main__":
    N,feature_dim = 2,10

    update = Update(feature_dim)

    s = torch.normal(0, 1, (N, 1, feature_dim))
    v = torch.normal(0, 1, (N, 3, feature_dim))

    pred_v, pred_s = update(v, s)

    assert pred_v.shape == (N, 3, feature_dim) and pred_s.shape == (N, 1, feature_dim)
