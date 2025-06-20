import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius_graph
from torch_geometric.nn import MessagePassing


class CosineCutoff(nn.Module):
    
    def __init__(self, cutoff):
        super(CosineCutoff, self).__init__()
        
        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2)


class GaussianSmearing(nn.Module):
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=True):
        super(GaussianSmearing, self).__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(0, self.cutoff, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


rbf_class_mapping = {"gauss": GaussianSmearing, "expnorm": ExpNormalSmearing}


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


act_class_mapping = {"ssp": ShiftedSoftplus, "silu": nn.SiLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "swish": Swish}


class Sphere(nn.Module):
    
    def __init__(self, l=2):
        super(Sphere, self).__init__()
        self.l = l
        
    def forward(self, edge_vec):
        edge_sh = self._spherical_harmonics(self.l, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2])
        return edge_sh
        
    @staticmethod
    def _spherical_harmonics(lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

        sh_1_0, sh_1_1, sh_1_2 = x, y, z
        
        if lmax == 1:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2], dim=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4], dim=-1)


class VecLayerNorm(nn.Module):
    def __init__(self, hidden_channels, trainable, norm_type="max_min"):
        super(VecLayerNorm, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.eps = 1e-12
        
        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight)
        
        if norm_type == "rms":
            self.norm = self.rms_norm
        elif norm_type == "max_min":
            self.norm = self.max_min_norm
        else:
            self.norm = self.none_norm
        
        self.reset_parameters()

    def reset_parameters(self):
        weight = torch.ones(self.hidden_channels)
        self.weight.data.copy_(weight)
    
    def none_norm(self, vec):
        return vec
        
    def rms_norm(self, vec):
        # vec: (num_atoms, 3 or 5, hidden_channels)
        dist = torch.norm(vec, dim=1)
        
        if (dist == 0).all():
            return torch.zeros_like(vec)
        
        dist = dist.clamp(min=self.eps)
        dist = torch.sqrt(torch.mean(dist ** 2, dim=-1))
        return vec / F.relu(dist).unsqueeze(-1).unsqueeze(-1)
    
    def max_min_norm(self, vec):
        # vec: (num_atoms, 3 or 5, hidden_channels)
        dist = torch.norm(vec, dim=1, keepdim=True)
        
        if (dist == 0).all():
            return torch.zeros_like(vec)
        
        dist = dist.clamp(min=self.eps)
        direct = vec / dist
        
        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        delta = (max_val - min_val).view(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)
        
        return F.relu(dist) * direct

    def forward(self, vec):
        # vec: (num_atoms, 3 or 8, hidden_channels)
        if vec.shape[1] == 3:
            vec = self.norm(vec)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.shape[1] == 8:
            vec1, vec2 = torch.split(vec, [3, 5], dim=1)
            vec1 = self.norm(vec1)
            vec2 = self.norm(vec2)
            vec = torch.cat([vec1, vec2], dim=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("VecLayerNorm only support 3 or 8 channels")


class Distance(nn.Module):
    def __init__(self, cutoff, max_num_neighbors=32, loop=True):
        super(Distance, self).__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.loop = loop

    def forward(self, pos, batch):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, loop=self.loop, max_num_neighbors=self.max_num_neighbors)
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.loop:
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        return edge_index, edge_weight, edge_vec


class NeighborEmbedding(MessagePassing):
    def __init__(self, hidden_channels, num_rbf, cutoff, max_z=100):
        super(NeighborEmbedding, self).__init__(aggr="add")
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
        # remove self loops
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        # propagate_type: (x: Tensor, W: Tensor)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W, size=None)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W

    
class EdgeEmbedding(MessagePassing):
    
    def __init__(self, num_rbf, hidden_channels):
        super(EdgeEmbedding, self).__init__(aggr=None)
        self.edge_proj = nn.Linear(num_rbf, hidden_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.edge_proj.weight)
        self.edge_proj.bias.data.fill_(0)
        
    def forward(self, edge_index, edge_attr, x):
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out
    
    def message(self, x_i, x_j, edge_attr):
        return (x_i + x_j) * self.edge_proj(edge_attr)
    
    def aggregate(self, features, index):
        # no aggregate
        return features


class TwoBodyAttn(nn.Module):
    def __init__(self, hidden_channels, num_heads):
        super(TwoBodyAttn, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        # learnable parameters
        self.feat1_2_Q_proj_in = nn.Linear(hidden_channels, hidden_channels)
        self.feat2_2_KV_proj_in = nn.Linear(hidden_channels, hidden_channels * 2)
        self.feat2_2_EG_proj_in = nn.Linear(hidden_channels, num_heads * 2)
        self.n_n_layernorm = nn.LayerNorm(hidden_channels, eps=1e-3)

    def reset_parameters(self):
        self.n_n_layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.feat1_2_Q_proj_in.weight)
        self.feat1_2_Q_proj_in.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.feat2_2_KV_proj_in.weight)
        self.feat2_2_KV_proj_in.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.feat2_2_EG_proj_in.weight)
        self.feat2_2_EG_proj_in.bias.data.fill_(0)

    def forward(self, feat):
        Q_in = self.feat1_2_Q_proj_in(feat)
        K_in, V_in = self.feat2_2_KV_proj_in(feat).chunk(2, dim=-1)
        E_in, G_in = self.feat2_2_EG_proj_in(feat).chunk(2, dim=-1)

        Q_in = Q_in.view(Q_in.shape[0], self.head_dim, self.num_heads)   # [num_nodes, hidden_channels] -> [num_nodes, head_dim, num_heads]
        K_in = K_in.view(K_in.shape[0], self.head_dim, self.num_heads)
        V_in = V_in.view(V_in.shape[0], self.head_dim, self.num_heads)

        _scale_factor = self.head_dim ** -0.5
        Q_in = Q_in * _scale_factor
        H_in = torch.einsum('idh, jdh->ijh', Q_in, K_in) + E_in
        gates_in = torch.sigmoid(G_in)
        A_in = torch.softmax(H_in, dim=0) * gates_in
        Va_in = torch.einsum('ijh, jdh->idh', A_in, V_in)     # [num_nodes, head_dim, num_heads]
        Va_in = Va_in.flatten(start_dim=1)  # [num_nodes, head_dim * num_heads]

        Va_in = self.n_n_layernorm(Va_in)
        if torch.isnan(Va_in).any():
            raise ValueError("after n_n_layernorm Va_in contains nan")
        return Va_in
    

class ManyBodyAttn(nn.Module):
    def __init__(self, hidden_channels, num_heads):
        super(ManyBodyAttn, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        # learnable parameters
        self.feat1_2_Q_proj_in = nn.Linear(hidden_channels, hidden_channels)
        self.feat2_2_KV_proj_in = nn.Linear(hidden_channels, hidden_channels * 2)
        self.feat2_2_EG_proj_in = nn.Linear(hidden_channels, num_heads * 2)
        self.n_n_layernorm = nn.LayerNorm(hidden_channels, eps=1e-3)

    def reset_parameters(self):
        self.n_n_layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.feat1_2_Q_proj_in.weight)
        self.feat1_2_Q_proj_in.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.feat2_2_KV_proj_in.weight)
        self.feat2_2_KV_proj_in.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.feat2_2_EG_proj_in.weight)
        self.feat2_2_EG_proj_in.bias.data.fill_(0)

    def forward(self, feat1, feat2):
        Q_in = self.feat1_2_Q_proj_in(feat1)    # ij
        K_in, V_in = self.feat2_2_KV_proj_in(feat2).chunk(2, dim=-1)    # jk
        E_in, G_in = self.feat2_2_EG_proj_in(feat2).chunk(2, dim=-1)    # ik

        Q_in = Q_in.view(Q_in.shape[0], Q_in.shape[1], self.head_dim, self.num_heads)   # [num_nodes, num_nodes, hidden_channels] -> [num_nodes, num_nodes, head_dim, num_heads]
        K_in = K_in.view(K_in.shape[0], K_in.shape[1], self.head_dim, self.num_heads)
        V_in = V_in.view(V_in.shape[0], V_in.shape[1], self.head_dim, self.num_heads)

        _scale_factor = self.head_dim ** -0.5
        Q_in = Q_in * _scale_factor
        H_in = torch.einsum('ijdh, jkdh->ijkh', Q_in, K_in) + E_in
        gates_in = torch.sigmoid(G_in)
        A_in = torch.softmax(H_in, dim=2) * gates_in
        # A_in = torch.softmax(H_in, dim=0) * gates_in
        # A_in = torch.softmax(H_in, dim=0)
        Va_in = torch.einsum('ijkh, jkdh->ijdh', A_in, V_in)     # [num_nodes, num_nodes, head_dim, num_heads]
        Va_in = Va_in.flatten(start_dim=2)  # [num_nodes, num_nodes, head_dim * num_heads]

        Va_in = self.n_n_layernorm(Va_in)
        if torch.isnan(Va_in).any():
            raise ValueError("after n_n_layernorm Va_in contains nan")
        return Va_in


class TwoBodyPadAttn(nn.Module):
    def __init__(self, hidden_channels, num_heads):
        super(TwoBodyPadAttn, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        self.feat1_2_Q_proj_in = nn.Linear(hidden_channels, hidden_channels)
        self.feat2_2_KV_proj_in = nn.Linear(hidden_channels, hidden_channels * 2)
        self.feat2_2_EG_proj_in = nn.Linear(hidden_channels, num_heads * 2)
        self.n_n_layernorm = nn.LayerNorm(hidden_channels, eps=1e-3)
        self.reset_parameters()

    def reset_parameters(self):
        self.n_n_layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.feat1_2_Q_proj_in.weight)
        self.feat1_2_Q_proj_in.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.feat2_2_KV_proj_in.weight)
        self.feat2_2_KV_proj_in.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.feat2_2_EG_proj_in.weight)
        self.feat2_2_EG_proj_in.bias.data.fill_(0)

    def forward(self, feat, mask=None):
        Q_in = self.feat1_2_Q_proj_in(feat)
        K_in, V_in = self.feat2_2_KV_proj_in(feat).chunk(2, dim=-1)
        E_in, G_in = self.feat2_2_EG_proj_in(feat).unsqueeze(1).chunk(2, dim=-1)

        Q_in = Q_in.view(Q_in.shape[0], Q_in.shape[1], self.head_dim, self.num_heads)   # [batchsize, num_nodes, head_dim, num_heads]
        K_in = K_in.view(K_in.shape[0], K_in.shape[1], self.head_dim, self.num_heads)
        V_in = V_in.view(V_in.shape[0], V_in.shape[1], self.head_dim, self.num_heads)

        _scale_factor = self.head_dim ** -0.5
        Q_in = Q_in * _scale_factor
        H_in = torch.einsum('bidh, bjdh->bijh', Q_in, K_in) + E_in

        if mask is not None:    # [batchsize, num_nodes, num_nodes]
            expanded_mask = mask.unsqueeze(-1).expand(-1, -1, -1, H_in.size(3))   # [batchsize, num_nodes, num_nodes, num_heads]
            H_in = H_in.masked_fill(~expanded_mask, float('-inf'))
    
        gates_in = torch.sigmoid(G_in)
        A_in = torch.softmax(H_in, dim=0) * gates_in
        Va_in = torch.einsum('bijh, bjdh->bidh', A_in, V_in)     # [batch_size, num_nodes, head_dim, num_heads]
        Va_in = Va_in.flatten(start_dim=2)  # [batch_size, num_nodes, head_dim * num_heads]

        Va_in = self.n_n_layernorm(Va_in)
        return Va_in


class ManyBodyPadAttn(nn.Module):
    def __init__(self, hidden_channels, num_heads):
        super(ManyBodyPadAttn, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        self.feat1_2_Q_proj_in = nn.Linear(hidden_channels, hidden_channels)
        self.feat2_2_KV_proj_in = nn.Linear(hidden_channels, hidden_channels * 2)
        self.feat2_2_EG_proj_in = nn.Linear(hidden_channels, num_heads * 2)
        self.n_n_layernorm = nn.LayerNorm(hidden_channels, eps=1e-3)
        self.reset_parameters()

    def reset_parameters(self):
        self.n_n_layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.feat1_2_Q_proj_in.weight)
        self.feat1_2_Q_proj_in.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.feat2_2_KV_proj_in.weight)
        self.feat2_2_KV_proj_in.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.feat2_2_EG_proj_in.weight)
        self.feat2_2_EG_proj_in.bias.data.fill_(0)

    def forward(self, feat1, feat2, mask=None):
        Q_in = self.feat1_2_Q_proj_in(feat1)    # ij
        K_in, V_in = self.feat2_2_KV_proj_in(feat2).chunk(2, dim=-1)    # jk
        E_in, G_in = self.feat2_2_EG_proj_in(feat2).unsqueeze(1).chunk(2, dim=-1)    # ik

        Q_in = Q_in.view(Q_in.shape[0], Q_in.shape[1], Q_in.shape[2], self.head_dim, self.num_heads)   # [batchsize, num_nodes, num_nodes, head_dim, num_heads]
        K_in = K_in.view(K_in.shape[0], K_in.shape[1], K_in.shape[2], self.head_dim, self.num_heads)
        V_in = V_in.view(V_in.shape[0], V_in.shape[1], V_in.shape[2], self.head_dim, self.num_heads)

        _scale_factor = self.head_dim ** -0.5
        Q_in = Q_in * _scale_factor
        H_in = torch.einsum('bijdh, bjkdh->bijkh', Q_in, K_in) + E_in

        if mask is not None:
            if mask.dim() == 4:  # [batch_size, num_nodes, num_nodes, num_nodes]
                expanded_mask = mask.unsqueeze(-1).expand(-1, -1, -1, -1, H_in.size(4))
            elif mask.dim() == 3:  # [batch_size, num_nodes, num_nodes]
                expanded_mask = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H_in.size(3), H_in.size(4))
            # H_in = H_in.masked_fill(~expanded_mask, float('-inf'))
            H_in = H_in.masked_fill(~expanded_mask, torch.finfo(mask.float().dtype).min)
    
        gates_in = torch.sigmoid(G_in)
        A_in = torch.softmax(H_in, dim=3) * gates_in
        # A_in = torch.softmax(H_in, dim=0) * gates_in
        # A_in = torch.softmax(H_in, dim=0)
        Va_in = torch.einsum('bijkh, bjkdh->bijdh', A_in, V_in)     # [batch_size, num_nodes, num_nodes, head_dim, num_heads]
        Va_in = Va_in.flatten(start_dim=3)  # [batch_size, num_nodes, num_nodes, head_dim * num_heads]

        Va_in = self.n_n_layernorm(Va_in)
        return Va_in


def calculate_max_edges(max_num_edges_save, is_bidirec):
    if is_bidirec:
        return max_num_edges_save // 2
    else:
        return max_num_edges_save

    
def get_feats(x, edge_index, r_ij, f_ij, cutoff_pruning, max_num_edges_save):
    non_self_loop_mask = edge_index[0] != edge_index[1]
    pruning_mask = (r_ij < cutoff_pruning) & non_self_loop_mask
    pruned_indices = r_ij[pruning_mask].argsort()[:max_num_edges_save]
    edge_index_pruning_mask = torch.zeros(r_ij.size(0), dtype=torch.bool, device=r_ij.device)
    edge_index_pruning_mask[pruning_mask.nonzero(as_tuple=False).squeeze(1)[pruned_indices]] = True
    f_ij_pruning = f_ij[edge_index_pruning_mask]

    x_x = torch.einsum('id, jd->ijd', x, x)   # [batchsize * num_nodes, batchsize * num_nodes, hidden_channels]

    if max_num_edges_save > 0:
        x_e = torch.einsum('id, jd->ijd', x, f_ij_pruning)  # [batchsize * num_nodes, batchsize * num_edges, hidden_channels]
        return x_x, x_e
    else:
        return x_x, None
    
def get_feats_with_padding(x, edge_index, r_ij, f_ij, cutoff_pruning, max_num_edges_save, hidden_channels, batch):
    batch_size = batch.max().item() + 1
    num_nodes_per_sample = torch.bincount(batch)
    max_num_nodes = num_nodes_per_sample.max().item()

    x_x = torch.zeros(batch_size, max_num_nodes, max_num_nodes, hidden_channels, device=x.device)
    if max_num_edges_save > 0:
        x_e = torch.zeros(batch_size, max_num_nodes, max_num_edges_save, hidden_channels, device=x.device)

    mask_xxx = torch.zeros(batch_size, max_num_nodes, max_num_nodes, max_num_nodes, device=x.device, dtype=torch.bool)
    mask_xxe = torch.zeros(batch_size, max_num_nodes, max_num_nodes, device=x.device, dtype=torch.bool)

    for b in range(batch_size):
        num_nodes = num_nodes_per_sample[b].item()
        x_b = x[batch == b]
        x_x[b, :num_nodes, :num_nodes] = torch.einsum('id,jd->ijd', x_b, x_b)
        mask_xxx[b, :num_nodes, :num_nodes, :num_nodes] = 1
        mask_xxe[b, :num_nodes, :num_nodes] = 1

        if max_num_edges_save > 0:
            mask_edges = (batch[edge_index[0]] == b) & (batch[edge_index[1]] == b)
            edge_index_b = edge_index[:, mask_edges]
            f_ij_b = f_ij[mask_edges]
            r_ij_b = r_ij[mask_edges]

            non_self_loop_mask = edge_index_b[0] != edge_index_b[1]
            edge_index_b = edge_index_b[:, non_self_loop_mask]
            f_ij_b = f_ij_b[non_self_loop_mask]
            r_ij_b = r_ij_b[non_self_loop_mask]

            pruning_mask = r_ij_b < cutoff_pruning
            edge_index_b = edge_index_b[:, pruning_mask]
            f_ij_b = f_ij_b[pruning_mask]
            r_ij_b = r_ij_b[pruning_mask]

            sorted_indices = r_ij_b.argsort()
            edge_index_b = edge_index_b[:, sorted_indices]
            f_ij_b = f_ij_b[sorted_indices]
            
            if f_ij_b.size(0) < max_num_edges_save:
                f_ij_b = torch.cat([f_ij_b, torch.zeros((max_num_edges_save - f_ij_b.size(0), f_ij_b.size(1)), device=f_ij_b.device)], dim=0)
            x_e[b, :num_nodes, :max_num_edges_save] = torch.einsum('id,jd->ijd', x_b, f_ij_b[:max_num_edges_save])

    if max_num_edges_save > 0:
        return x_x, x_e, mask_xxx, mask_xxe
    else:
        return x_x, None, mask_xxx, mask_xxe

