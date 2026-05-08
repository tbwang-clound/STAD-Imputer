"""
STIMP (graphdiffusion) model adapted for traffic / air-quality benchmarks.

Original: STIMP-release/model/graphdiffusion.py
Adaptation: Added traffic_mode to SpatialTemporalEncoding so it works
            without coastal geographic files (is_sea.npy, mean.npy, std.npy).

In traffic mode:
  - No is_sea mask (all nodes are active)
  - Grid-based position embedding is replaced by a learnable node embedding
  - GCN spatial encoding is retained and fed the learnable position embedding
  - All other logic (temporal encoding, residual connection, diffusion) is unchanged
"""

import numpy as np
import torch
from torch import nn
from linear_attention_transformer import LinearAttentionTransformer
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from torch.nn import Parameter
import os


class IAP_base(nn.Module):
    """STIMP diffusion imputer (single layer with GCN spatial + temporal encoding)."""

    def __init__(self, config, low_bound, high_bound):
        super().__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.num_steps = self.config.num_steps

        self.diffusion_model = SpatialTemporalEncoding(config=config, low_bound=low_bound, high_bound=high_bound)

        if config.schedule == "quad":
            self.beta = torch.linspace(
                config.beta_start ** 0.5, config.beta_end ** 0.5, self.num_steps
            ) ** 2
        elif config.schedule == "linear":
            self.beta = torch.linspace(
                config.beta_start, config.beta_end, self.num_steps
            )
        self.alpha_hat = 1 - self.beta
        self.alpha = torch.cumprod(self.alpha_hat, dim=0)
        self.alpha_prev = F.pad(self.alpha[:-1], (1, 0), value=1.)
        self.alpha_torch = self.alpha.float().to(self.device).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        self.low_bound = low_bound
        self.high_bound = high_bound

    def get_randmask(self, observed_mask, sample_ratio):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = sample_ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def forward(self, observed_data):
        observed_mask = torch.ones_like(observed_data, device=self.device)
        adj = torch.ones((observed_mask.shape[-1], observed_mask.shape[-1]), device=self.device)
        is_train = 1
        return self.trainstep(observed_data, observed_mask, adj, is_train)

    def trainstep(self, observed_data, observed_mask, adj, is_train, set_t=-1):
        cond_mask = self.get_randmask(observed_mask, self.config.missing_ratio)
        cond_mask = cond_mask.to(self.device)
        B = observed_data.shape[0]

        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.config.num_steps, [B]).to(self.device)

        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(observed_data)

        mean = (observed_data * cond_mask).sum(dim=1, keepdim=True) / (cond_mask.sum(dim=1, keepdim=True) + 1e-5)
        mean_ = mean.expand_as(observed_data)

        observed_data_imputed = torch.where(cond_mask.bool(), observed_data, mean_)
        noisy_data = (current_alpha ** 0.5) * observed_data_imputed + (1.0 - current_alpha) ** 0.5 * noise

        total_input = torch.stack([observed_data_imputed, (1 - cond_mask) * noisy_data], dim=3)
        predicted = self.diffusion_model(total_input, cond_mask, adj, t)

        target_mask = observed_mask - cond_mask
        residual = (observed_data - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

        return loss

    def impute(self, observed_data, observed_mask, adj, n_samples):
        B, T, K, N = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, T, K, N)
        mean = (observed_data * observed_mask).sum(1, keepdim=True) / (observed_mask.sum(1, keepdim=True) + 1e-5)
        mean_ = mean.expand_as(observed_data)

        with torch.no_grad():
            for i in range(n_samples):
                current_sample = torch.randn_like(observed_data).to(self.device) + mean_
                observed_data_imputed = torch.where(observed_mask.bool(), observed_data, mean.expand_as(observed_data))

                for t in range(self.num_steps - 1, -1, -1):
                    noisy_target = current_sample
                    total_input = torch.stack([observed_data_imputed, (1 - observed_mask) * noisy_target], dim=3)
                    predicted = self.diffusion_model(total_input, observed_mask, adj,
                                                     (torch.ones(B) * t).long().to(self.device))
                    coeff1 = (1 - self.alpha_prev[t]) * (self.alpha_hat[t]) ** 0.5 / (1 - self.alpha[t])
                    coeff2 = ((1 - self.alpha_hat[t]) * (self.alpha_prev[t]) ** 0.5) / (1 - self.alpha[t])
                    current_sample = coeff1 * current_sample + coeff2 * predicted
                    if t > 0:
                        noise = torch.randn_like(current_sample)
                        sigma = (
                            (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                        ) ** 0.5
                        current_sample += sigma * noise

                imputed_samples[:, i] = current_sample.detach().cpu()
        return imputed_samples


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class SpatialTemporalEncoding(nn.Module):
    """STIMP inner layer: input projection -> temporal -> GCN spatial -> output projection."""

    def __init__(self, config, low_bound, high_bound):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # ---- Traffic mode detection ----
        self.traffic_mode = getattr(config, 'num_nodes', None) is not None

        self.input_projection = Conv1d_with_init(2, config.hidden_channels, 1)
        self.mid_projection = Conv1d_with_init(config.hidden_channels, 2 * config.hidden_channels, 1)
        self.output_projection = Conv1d_with_init(config.hidden_channels, 1, 1)

        self.spatial_encoding = GCN(
            self.config.hidden_channels, self.config.hidden_channels,
            self.config.hidden_channels // 4, 3
        )

        self.time_encoding = LinearAttentionTransformer(
            dim=self.config.hidden_channels, depth=1, heads=1,
            max_seq_len=16, n_local_attn_heads=0, local_attn_window_size=0
        )
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config.num_steps,
            embedding_dim=config.diffusion_embedding_size,
            projection_dim=config.hidden_channels
        )

        self.norm1 = nn.LayerNorm(self.config.hidden_channels)
        self.norm2 = nn.LayerNorm(self.config.hidden_channels)

        if not self.traffic_mode:
            # Coastal mode: load geographic files
            self.is_sea = torch.from_numpy(np.load(os.path.join(self.config.data_root, self.config.area, "is_sea.npy"))).to(self.device)
            self.mean = torch.from_numpy(np.load(os.path.join(self.config.data_root, self.config.area, "mean.npy"))).to(self.device)
            self.std = torch.from_numpy(np.load(os.path.join(self.config.data_root, self.config.area, "std.npy"))).to(self.device)
            self.is_sea = self.is_sea.bool()
            learnable_position_embedding = self.get_position_embeding()[:, self.is_sea]
            self.register_buffer("embedding", learnable_position_embedding.float())
            self.projection1 = nn.Linear(3, config.hidden_channels // 4)
        else:
            # Traffic mode: learnable node embedding replaces grid position embedding
            N = config.num_nodes
            self.node_emb = nn.Embedding(N, config.hidden_channels // 4)
            nn.init.xavier_uniform_(self.node_emb.weight)
            self.register_buffer("embedding", torch.zeros(1, N))
            self.projection1 = nn.Identity()

        self.gn = nn.GroupNorm(4, config.hidden_channels)

        self.low_bound = low_bound
        self.high_bound = high_bound

    def forward(self, x, mask, adj, diffusion_step):
        B, T, K, C, N = x.shape

        # ---- Input projection ----
        x = rearrange(x, 'b t k c n->(b t k) c n')
        x = self.input_projection(x)
        C = x.shape[1]
        x = rearrange(x, '(b t k) c n -> b t k c n', b=B, t=T, k=K)

        # ---- Diffusion step embedding ----
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        diffusion_emb = diffusion_emb.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        input = x + diffusion_emb

        # ---- Temporal encoding ----
        x = rearrange(input, 'b t k c n->(b k n) t c')
        x = self.time_encoding(x)
        x = rearrange(x, '(b k n) t c->(b k t) n c', b=B, n=N, k=K)

        # ---- Spatial encoding ----
        x_in = x
        position_embedding = self.get_position_embedding_()
        x = self.spatial_encoding(x, adj, position_embedding)

        x = rearrange(x, '(b k t) n c-> (b t k) c n', b=B, k=K, t=T)
        x_in = rearrange(x_in, '(b k t) n c-> (b t k) c n', b=B, k=K, t=T)
        x = x + x_in
        x = self.gn(x)

        # ---- Output projection ----
        x = self.mid_projection(x)
        gate, filter = torch.chunk(x, 2, dim=1)
        x = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(x)
        y = rearrange(y, '(b t k) c n -> b t k c n', b=B, k=K, t=T)
        y = y.squeeze(3)

        low_bound = self.low_bound.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(y)
        high_bound = self.high_bound.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(y)
        y = torch.clamp(y, low_bound, high_bound)
        return y

    def get_position_embeding(self):
        """Grid-based position embedding (coastal mode only)."""
        height = self.config.height
        width = self.config.width
        pos_w = torch.arange(0., width) / width
        pos_h = torch.arange(0., height) / height
        pos_w = pos_w.unsqueeze(0).expand(height, -1)
        pos_h = pos_h.unsqueeze(1).expand(-1, width)
        pe = torch.stack([self.mean.cpu(), self.std.cpu(), pos_h], 0)
        pe = pe.to(self.device)
        return pe

    def get_position_embedding_(self):
        """Return position embedding tensor of shape (hidden//4, N)."""
        if self.traffic_mode:
            idx = torch.arange(self.config.num_nodes, device=self.node_emb.weight.device)
            emb = self.node_emb(idx)          # (N, hidden//4)
            return emb.transpose(0, 1)        # (hidden//4, N)

        x = self.embedding.transpose(0, 1)
        x = self.projection1(x)
        x = x.transpose(0, 1)
        return x


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = F.silu(self.projection1(x))
        x = F.silu(self.projection2(x))
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class GCN(nn.Module):
    def __init__(self, c_in, c_out, c_hid, num_types, temp=1):
        super().__init__()
        self.linear = nn.Linear(c_in, c_out, bias=False)
        self.num_types = num_types
        self.temp = temp

        nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        self.weights_pool = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(c_hid, c_in, c_out)))

    def forward(self, node_feats, adj_matrix, position_embedding):
        node_feats = torch.matmul(adj_matrix, node_feats)
        position_embedding = torch.matmul(adj_matrix, position_embedding.transpose(0, 1))
        position_weights = torch.einsum('nd, dio-> nio', position_embedding, self.weights_pool)
        node_feats = torch.einsum('bni, nio->bno', node_feats, position_weights)
        return node_feats
