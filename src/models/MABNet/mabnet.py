from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from src.models.MABNet.utils import *


class MABNet(nn.Module):

    def __init__(
        self,
        lmax=2,
        vecnorm_type='none',
        trainable_vecnorm=False,
        num_heads=8,
        num_layers=9,
        hidden_channels=256,
        num_rbf=32,
        rbf_type="expnorm",
        trainable_rbf=False,
        activation="silu",
        attn_activation="silu",
        max_z=100,
        cutoff=5.0,
        cutoff_pruning=1.6,
        max_num_neighbors=32,
        max_num_edges_save=32,
        use_padding=True,
        many_body=True,
    ):
        super(MABNet, self).__init__()
        self.lmax = lmax
        self.vecnorm_type = vecnorm_type
        self.trainable_vecnorm = trainable_vecnorm
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.max_z = max_z
        self.cutoff = cutoff
        self.cutoff_pruning = cutoff_pruning
        self.max_num_neighbors = max_num_neighbors
        self.max_num_edges_save = max_num_edges_save
        self.use_padding = use_padding
        self.many_body = many_body

        # learnable parameters
        self.embedding = nn.Embedding(max_z, hidden_channels)   # scalar embedding
        self.distance = Distance(cutoff, max_num_neighbors=max_num_neighbors, loop=True)
        self.sphere = Sphere(l=lmax)
        self.distance_expansion = rbf_class_mapping[rbf_type](cutoff, num_rbf, trainable_rbf)
        self.neighbor_embedding = NeighborEmbedding(hidden_channels, num_rbf, cutoff, max_z).jittable()
        self.edge_embedding = EdgeEmbedding(num_rbf, hidden_channels).jittable()

        self.mp_layers = nn.ModuleList()
        mp_kwargs = dict(
            num_heads=num_heads, 
            hidden_channels=hidden_channels, 
            activation=activation, 
            attn_activation=attn_activation, 
            cutoff=cutoff, 
            cutoff_pruning=cutoff_pruning,
            max_num_edges_save=max_num_edges_save,
            vecnorm_type=vecnorm_type, 
            trainable_vecnorm=trainable_vecnorm,
            use_padding=use_padding,
        )

        for _ in range(num_layers - 1):
            layer = ManybodyMPLayer(last_layer=False, **mp_kwargs).jittable()
            self.mp_layers.append(layer)
        self.mp_layers.append(ManybodyMPLayer(last_layer=True, **mp_kwargs).jittable())

        self.out_norm = nn.LayerNorm(hidden_channels)
        self.vec_out_norm = VecLayerNorm(hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type)
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        self.neighbor_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        for layer in self.mp_layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()
        self.vec_out_norm.reset_parameters()
        
    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        """
        Args:
            data (Data): Input data containing:
                - z (Tensor): Atomic numbers, shape [num_nodes].
                - pos (Tensor): Node positions (coordinates), shape [num_nodes, 3].
                - batch (Tensor): Batch indices for each node, shape [num_nodes].

        Returns:
            Tuple[Tensor, Tensor]: A tuple of:
                - x (Tensor): Updated node scalar features, shape [num_nodes, hidden_channels].
                - vec (Tensor): Updated node vector features, shape [num_nodes, ((lmax + 1)^2 - 1), hidden_channels].
        """

        z, pos, batch = data.z, data.pos, data.batch
        
        # Embedding Layers
        x = self.embedding(z)   # [node_nums, hidden_channels]
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)    # [edge_nums, 32]
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        edge_vec = self.sphere(edge_vec)    # torch.Size([5176, 8])

        x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)
        vec = torch.zeros(x.size(0), ((self.lmax + 1) ** 2) - 1, x.size(1), device=x.device)    # vector embedding
        edge_attr = self.edge_embedding(edge_index, edge_attr, x)   # [edge_nums, hidden_channels]
        
        # MP Layers
        for i, attn in enumerate(self.mp_layers[:-1]):
            dx, dvec, dedge_attr = attn(x, vec, edge_index, edge_weight, edge_attr, edge_vec, batch)
            x = x + dx  # [node_nums, hidden_channels]
            vec = vec + dvec
            edge_attr = edge_attr + dedge_attr

        dx, dvec, _ = self.mp_layers[-1](x, vec, edge_index, edge_weight, edge_attr, edge_vec, batch)
        x = x + dx
        vec = vec + dvec
        
        x = self.out_norm(x)
        vec = self.vec_out_norm(vec)

        return x, vec


class ManybodyMPLayer(MessagePassing):
    def __init__(
        self,
        num_heads,
        hidden_channels,
        activation,
        attn_activation,
        cutoff,
        cutoff_pruning,
        max_num_edges_save,
        vecnorm_type,
        trainable_vecnorm,
        last_layer=False,
        use_padding=True,
        is_bidirec=True,
    ):
        super(ManybodyMPLayer, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer
        self.cutoff_pruning = cutoff_pruning
        self.max_num_edges_save = calculate_max_edges(max_num_edges_save, is_bidirec)
        self.use_padding = use_padding

        # learnable parameters
        self.layernorm = nn.LayerNorm(hidden_channels)
        self.vec_layernorm = VecLayerNorm(hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type)
        self.act = act_class_mapping[activation]()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff)
        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 2, bias=False)
        self.dv_proj = nn.Linear(hidden_channels, hidden_channels)

        self.mabybody_attention_three_body = ManyBodyPadAttn(hidden_channels, num_heads)
        if self.max_num_edges_save > 0:
            self.manybody_attention_four_body = ManyBodyPadAttn(hidden_channels, num_heads)

        if self.max_num_edges_save > 0:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )

        self.s_proj = nn.Linear(hidden_channels, hidden_channels * 2)
        if not self.last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels)
            self.w_src_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.w_trg_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        
        self.reset_parameters()
        
    @staticmethod
    def vector_rejection(vec, d_ij):
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        self.vec_layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.s_proj.weight)
        self.s_proj.bias.data.fill_(0)
        
        if not self.last_layer:
            nn.init.xavier_uniform_(self.f_proj.weight)
            self.f_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.w_src_proj.weight)
            nn.init.xavier_uniform_(self.w_trg_proj.weight)

        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.dv_proj.weight)
        self.dv_proj.bias.data.fill_(0)

        
    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij, batch):
        """
        Args:
            x (Tensor): Scalar embedding of nodes, shape [batchsize * num_nodes, hidden_channels].
            vec (Tensor): Vector embedding of nodes, shape [batchsize * num_nodes, ((lmax + 1)^2 - 1), hidden_channels].
            edge_index (Tensor): Edge indices, shape [2, batchsize * num_edges].
            r_ij (Tensor): Edge distances, shape [batchsize * num_edges].
            f_ij (Tensor): Edge features, shape [batchsize * num_edges, hidden_channels].
            d_ij (Tensor): Edge directions, shape [batchsize * num_edges, ((lmax + 1)^2 - 1)].
            batch (Tensor): Batch indices for each node, shape [batchsize * num_nodes].

        Returns:
            Tuple[Tensor, Tensor, Optional[Tensor]]: A tuple containing:
                - dx (Tensor): Updated scalar embedding, shape [batchsize * num_nodes, hidden_channels].
                - dvec (Tensor): Updated vector embedding, shape [batchsize * num_nodes, ((lmax + 1)^2 - 1), hidden_channels].
                - df_ij (Optional[Tensor]): Updated edge features, shape [batchsize * num_edges, hidden_channels], or None if last layer.
        """

        x = self.layernorm(x)   # scalar embedding
        vec = self.vec_layernorm(vec)   # vector embedding

        batch_size = batch.max().item() + 1
        num_nodes_per_sample = torch.bincount(batch)
        x_x, x_e, mask_three_body, mask_four_body = get_feats_with_padding(
            x,
            edge_index,
            r_ij,
            f_ij,
            self.cutoff_pruning,
            self.max_num_edges_save,
            self.hidden_channels,
            batch,
        )
        v = self.mabybody_attention_three_body(x_x, x_x, mask_three_body).mean(dim=2)    # 3-body
        if self.max_num_edges_save > 0:
            v_four_body = self.manybody_attention_four_body(x_x, x_e, mask_four_body).mean(dim=2)   # 4-body
            v = torch.cat([v, v_four_body], dim=-1)
            v = self.mlp(v)
        
        v_list = []
        for b in range(batch_size):
            num_nodes = num_nodes_per_sample[b].item()
            v_b = v[b, :num_nodes]
            v_list.append(v_b)
        v = torch.cat(v_list, dim=0)

        v = v.reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)   # [batchsize * num_edges, num_heads * head_dim] -> [batchsize * num_edges, num_heads, head_dim]

        vec1, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_sum = vec1.sum(dim=1)

        # propagate_type: (v: Tensor, dv: Tensor, vec: Tensor, r_ij: Tensor, d_ij: Tensor)
        x, vec_out = self.propagate(
            edge_index,
            v=v,
            dv=dv,
            vec=vec,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        
        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_sum * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out

        if not self.last_layer:
            # edge_updater_type: (vec: Tensor, d_ij: Tensor, f_ij: Tensor)
            df_ij = self.edge_updater(edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij)
            return dx, dvec, df_ij
        else:
            return dx, dvec, None

    def message(self, v_j, vec_j, dv, r_ij, d_ij):
        """
        Args:
            v_j (Tensor): Value embeddings of neighboring nodes, shape [batchsize * num_edges, num_heads, head_dim].
            vec_j (Tensor): Vector embeddings of neighboring nodes, shape [batchsize * num_edges, ((lmax + 1)^2 - 1), hidden_channels].
            dv (Tensor): Projected edge features, shape [batchsize * num_edges, num_heads, head_dim].
            r_ij (Tensor): Edge distances, shape [batchsize * num_edges].
            d_ij (Tensor): Edge directions, shape [batchsize * num_edges, ((lmax + 1)^2 - 1)].

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - v_j (Tensor): Updated value embeddings, shape [batchsize * num_edges, hidden_channels].
                - vec_j (Tensor): Updated vector embeddings, shape [batchsize * num_edges, ((lmax + 1)^2 - 1), hidden_channels].
        """
        
        v_j = v_j * self.cutoff(r_ij).unsqueeze(1).unsqueeze(2)
        v_j = v_j * dv
        v_j = v_j.view(-1, self.hidden_channels)
        s1, s2 = torch.split(self.act(self.s_proj(v_j)), self.hidden_channels, dim=1)
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)
        """
        v_j: shape [batchsize * num_edges, hidden_channels]
        vec_j: shape [batchsize * num_edges, ((lmax + 1)^2 - 1), hidden_channels]
        """
        return v_j, vec_j
    
    def edge_update(self, vec_i, vec_j, d_ij, f_ij):
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        df_ij = self.act(self.f_proj(f_ij)) * w_dot
        return df_ij

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        """
        x: shape [batchsize * num_edges, hidden_channels]
        vec: shape [batchsize * num_edges, ((lmax + 1)^2 - 1), hidden_channels]
        """
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        """
        x: shape [batchsize * num_nodes, hidden_channels]
        vec: shape [batchsize * num_nodes, ((lmax + 1)^2 - 1), hidden_channels]
        """
        return x, vec

    def update(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs

