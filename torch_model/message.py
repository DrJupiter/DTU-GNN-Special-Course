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

#from PaiNN import segment_sum


import torch
from torch import nn
from torch_scatter import scatter_add, scatter


def min_max(a,b):
    return (a + b - torch.abs(a - b))/2

def radial_basis(radiual_range: torch.Tensor, norm_direction: torch.Tensor, cutoff_radius):
    return torch.sin( radiual_range * torch.pi / cutoff_radius * norm_direction ) / norm_direction

def f_cut(direction_embeddings, cutoff_radius):
    # B x F * 3 - 1 -> B x F * 3
    delta = direction_embeddings - cutoff_radius

    # B x F * 3 -> B x F * 3
    mask = min_max(delta, 0) * 1/delta

    # B x F * 3
    values = 0.5 * (
                torch.cos(torch.pi * direction_embeddings/cutoff_radius)
                + 1)

    # B x F * 3
    return values * mask

class Message(nn.Module):

    def __init__(self, feature_dim, n, cutoff_radius):
        super().__init__() 
        self.s_lin_1 = nn.Linear(feature_dim, feature_dim)
        self.s_lin_2 = nn.Linear(feature_dim, 3 * feature_dim)
        self.r_lin = nn.Linear(n, feature_dim * 3)

        self.radiual_range = nn.Parameter(torch.tensor(range(1, n+1)), requires_grad=False)
        self.cutoff_radius = cutoff_radius
        self.norm = torch.vmap(lambda x: (x @ x) ** 0.5)
        self.feature_dim = feature_dim
       
    def forward(self, vectors, scalars, directions, idx_i, idx_j):

        """
        vectors: A x 3 x F
        scalars: A x 1 x F
        directions: EDGES x 3
        """

        # phi
        # A x 1 x F -> A x 1 x F
        s = self.s_lin_1(scalars)

        s = torch.nn.functional.silu(s)

        # A x 1 x F -> A x 1 x F * 3
        s = self.s_lin_2(s)

        # W
        
        # EDGES x 3 -> EDGES x 1
        norm_r = self.norm(directions).reshape(-1, 1)

        # A -> B = B - A

        # EDGES x 1 -> EDGES x n
        r = radial_basis(self.radiual_range, norm_r, self.cutoff_radius)

        # EDGES x n -> EDGES x F * 3
        r = self.r_lin(r)

        # EDGES x F * 3 -> EDGES x F * 3
        r = f_cut(norm_r, self.cutoff_radius) * r

                #  EDGES x 1 x F * 3    *  EDGES x 1 x F * 3
        combined = s[idx_i]             *  r[:, None, :]

        # EDGES x 1 x F, EDGES x 1 x F, EDGES x 1 x F
        delta_s, delta_v1, delta_v2 = torch.split(combined, self.feature_dim, dim=-1)

                # A x 3 x F        * EDGES x 1 x F +    EDGES x 1 x F *  EDGES x 3 x 1 -> EDGES x 3 x F
        delta_v = vectors[idx_i]   * delta_v1      +    delta_v2      * (directions/norm_r)[:, :, None]
        # EDGES x 3 x F -> A x 3 x F

        # aggregate
        # EDGES x 1 x F -> A x 1 x F
        delta_s = scatter(delta_s, idx_j, dim=0)
        # EDGES x 3 x F -> A x 3 x F
        delta_v = scatter(delta_v, idx_j, dim=0)

        assert delta_v.shape == vectors.shape, f"{delta_v.shape} != {vectors.shape}"
        assert delta_s.shape == scalars.shape, f"{delta_s.shape} != {scalars.shape}"

        return delta_v, delta_s

if __name__ == "__main__":
    
    message = Message(10, 20, 3.)

    scalars = torch.normal(0, 1, (3, 1, 10))
    positions = torch.normal(0, 1, (3, 3))
    vectors = torch.normal(0, 1, (3, 3, 10))

    idx_i = torch.tensor([0, 0, 1, 2])
    idx_j = torch.tensor([1,2,0,0])

    dir_i_to_j = lambda x, i, j: x[j] - x[i]
    directions = dir_i_to_j(positions, idx_i, idx_j)

    delta_s, delta_v = message(vectors, scalars, directions, idx_i, idx_j)

