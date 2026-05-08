"""
STAD-Imputer Core Modules

This file contains the three adaptive modules proposed in STAD-Imputer:
  - ATI: Adaptive Temporal Integrator
  - ANA: Adaptive Neighborhood Aggregator
  - AHM: Adaptive Heterogeneity Modulator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple


# =============================================================================
# ATI: Adaptive Temporal Integrator
# =============================================================================
# Supporting components for ATI

class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable convolution used as a local TCN expert."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, causal=False):
        super().__init__()
        self.causal = causal
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) * dilation if causal else 'same'

        self.depthwise_conv = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=self.padding,
            dilation=dilation,
            groups=in_channels
        )

    def forward(self, x):
        out = self.depthwise_conv(x)
        if self.causal and isinstance(self.padding, int):
            out = out[:, :, :-self.padding]
        return out


class GlobalBottleneckExpert(nn.Module):
    """
    Global temporal context expert.
    Compresses T time steps into a summary vector and broadcasts it back.
    """

    def __init__(self, dim, num_latents=8, dropout=0.1):
        super().__init__()
        self.pool_query = nn.Linear(dim, num_latents)
        self.broadcast_ffn = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        B, C, T = x.shape
        x_seq = x.permute(0, 2, 1)  # (B, T, C)

        pool_scores = self.pool_query(x_seq)      # (B, T, M)
        pool_weights = F.softmax(pool_scores, dim=1)  # softmax over T
        summary = torch.einsum('btm,btc->bmc', pool_weights, x_seq)  # (B, M, C)

        pooled_summary = torch.mean(summary, dim=1)  # (B, C)
        global_context = pooled_summary.unsqueeze(1).expand(-1, T, -1)  # (B, T, C)

        combined = torch.cat([x_seq, global_context], dim=-1)  # (B, T, 2C)
        context = self.broadcast_ffn(combined)                  # (B, T, C)

        return (context + x_seq).permute(0, 2, 1)  # (B, C, T)


class AdaptiveTCNBlock(nn.Module):
    """
    One block of the Adaptive TCN with multi-dilation local experts and an
    optional global bottleneck expert, gated by a lightweight conv router.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout,
                 causal=False, dilation_choices=None,
                 gate_channels=32, gate_kernel_size=3):
        super().__init__()

        if dilation_choices is None:
            dilation_choices = [1, 2, 4, 8]
        self.dilation_choices = dilation_choices
        self.causal = causal
        self.gate_kernel_size = gate_kernel_size

        # Local dilated experts
        self.experts = nn.ModuleList()
        for d in dilation_choices:
            self.experts.append(
                DepthwiseSeparableConv1d(in_channels, in_channels, kernel_size,
                                        dilation=d, causal=causal)
            )

        # Global expert
        self.global_expert = GlobalBottleneckExpert(dim=in_channels, dropout=dropout)
        self.experts.append(self.global_expert)
        self.num_experts = len(self.experts)

        # Lightweight gating network
        self.gate_padding = (gate_kernel_size - 1) * 1 if causal else 0
        self.gate_conv = nn.Conv1d(
            in_channels, gate_channels, gate_kernel_size,
            padding=self.gate_padding if causal else 'same'
        )
        self.gate_activation = nn.ReLU()
        self.gate_output = nn.Conv1d(gate_channels, self.num_experts, 1)

        # Point-wise projection
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1)

        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / fan_in ** 0.5 if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)

        local_ctx = self.gate_conv(x)
        if self.causal:
            local_ctx = local_ctx[:, :, :-self.gate_padding]
        local_ctx = self.gate_activation(local_ctx)
        gating_weights = F.softmax(self.gate_output(local_ctx), dim=1)  # (B, E, T)

        expert_outputs = [e(x) for e in self.experts]
        stacked = torch.stack(expert_outputs, dim=1)  # (B, E, C, T)

        weighted = torch.einsum('bet,bect->bct', gating_weights, stacked)

        out = self.pointwise_conv(weighted)
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.dropout(self.relu(out))
        return out + res


class StackedAdaptiveTCN(nn.Module):
    """Stack of AdaptiveTCNBlocks."""

    def __init__(self, num_layers, in_channels, out_channels, hidden_channels,
                 kernel_size, dropout, causal=False,
                 dilation_choices=None, gate_channels=32, gate_kernel_size=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                AdaptiveTCNBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    causal=causal,
                    dilation_choices=dilation_choices,
                    gate_channels=gate_channels,
                    gate_kernel_size=gate_kernel_size
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ATI(nn.Module):
    """
    Adaptive Temporal Integrator (ATI).

    Integrates multi-scale dilated temporal branches with an adaptive gating
    mechanism to capture temporal patterns across multiple receptive fields.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden / output dimension.
        tcn_layers (int): Number of stacked TCN blocks.
        kernel_size (int): Convolution kernel size.
        mode (str): 'imputation' uses non-causal convolutions.
        dropout (float): Dropout probability.
        dilation_choices (list): List of dilation rates for local experts.
    """

    def __init__(self, input_dim, hidden_dim, tcn_layers=2, kernel_size=3,
                 mode='imputation', dropout=0.1, dilation_choices=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        causal = False if mode == 'imputation' else True

        self.adaptive_tcn = StackedAdaptiveTCN(
            num_layers=tcn_layers,
            in_channels=input_dim,
            out_channels=hidden_dim,
            hidden_channels=hidden_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=causal,
            dilation_choices=dilation_choices,
            gate_channels=hidden_dim,
            gate_kernel_size=kernel_size
        )

    def forward(self, x, B, N):
        """
        Args:
            x: (B*K*N, T, C)
            B: batch size
            N: number of spatial nodes
        Returns:
            x: (B*K*N, T, C)
        """
        x_tcn_in = x.permute(0, 2, 1)   # (B*K*N, C, T)
        x_tcn_out = self.adaptive_tcn(x_tcn_in)
        return x_tcn_out.permute(0, 2, 1)  # (B*K*N, T, C)


# =============================================================================
# ANA: Adaptive Neighborhood Aggregator
# =============================================================================

class ANA(nn.Module):
    """
    Adaptive Neighborhood Aggregator (ANA).

    Performs dual-path neighborhood aggregation based on:
      - Physical proximity (top-k neighbors from the adjacency matrix)
      - Prototype-guided semantic similarity (learnable prototypes)

    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output feature dimension.
        k_phys (int): Number of physical neighbors.
        k_feat (int): Number of semantic / feature-based neighbors.
        num_prototypes (int): Number of learnable prototype vectors.
        dropout (float): Dropout probability.
    """

    def __init__(self, in_dim, out_dim, k_phys=8, k_feat=8,
                 num_prototypes=32, dropout=0.1):
        super().__init__()
        self.out_dim = out_dim
        self.k_phys = k_phys
        self.k_feat = k_feat
        self.num_prototypes = num_prototypes

        self.fused_transform = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.SiLU()
        )
        self.gate_proj = nn.Sequential(
            nn.Linear(out_dim, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

        self.prototypes = nn.Parameter(torch.empty(num_prototypes, out_dim))

        self.out_proj = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.prototypes, mode='fan_in')

    def _compute_chunk_size(self, N):
        if N <= 1000:
            return N
        elif N <= 5000:
            return 1024
        else:
            return 2048

    def forward(self, node_feats, adj):
        """
        Args:
            node_feats: (B, N, in_dim)
            adj:        (N, N) adjacency matrix
        Returns:
            output: (B, N, out_dim)
        """
        B, N, _ = node_feats.shape
        device = node_feats.device

        h = self.fused_transform(node_feats)

        # Physical neighbors from adjacency matrix
        _, phys_neighbors = torch.topk(adj, self.k_phys, dim=-1)  # (N, k_phys)

        # Semantic neighbors via prototype matching
        h_norm = F.normalize(h, p=2, dim=-1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)
        node_proto_sim = torch.einsum('bnh,ph->bnp', h_norm, proto_norm)

        _, top_proto_idx = node_proto_sim.max(dim=-1)
        proto_node_sim = node_proto_sim.transpose(1, 2)
        _, feat_neighbors = torch.topk(proto_node_sim, self.k_feat, dim=-1)

        top_proto_idx_exp = top_proto_idx.unsqueeze(-1).expand(-1, -1, self.k_feat)
        dyn_feat_neighbors = feat_neighbors.gather(1, top_proto_idx_exp)

        combined_neighbors = torch.cat([
            phys_neighbors.unsqueeze(0).expand(B, -1, -1),
            dyn_feat_neighbors
        ], dim=-1)
        k_total = self.k_phys + self.k_feat

        # Chunked aggregation
        chunk_size = self._compute_chunk_size(N)
        outputs = []
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            neighbors_chunk = combined_neighbors[:, i:end]

            batch_offset = torch.arange(B, device=device).view(B, 1, 1) * N
            abs_indices = (neighbors_chunk + batch_offset).reshape(-1)

            h_flat = h.reshape(B * N, self.out_dim)
            h_j_chunk = h_flat[abs_indices].view(B, end - i, k_total, self.out_dim)

            gate_scores = self.gate_proj(h_j_chunk).squeeze(-1)
            gate_weights = self.dropout(gate_scores)

            h_prime_chunk = (h_j_chunk * gate_weights.unsqueeze(-1)).sum(dim=2) / \
                            (gate_weights.sum(dim=2, keepdim=True) + 1e-8)
            outputs.append(h_prime_chunk)

        h_prime = torch.cat(outputs, dim=1)
        return self.out_proj(h_prime + node_feats)


# =============================================================================
# AHM: Adaptive Heterogeneity Modulator
# =============================================================================

class AHM(nn.Module):
    """
    Adaptive Heterogeneity Modulator (AHM).

    Performs multi-scale spatial aggregation and applies node-wise sparse expert
    modulation to capture spatial heterogeneity.

    Args:
        hidden_dim (int): Feature dimension.
        pos_dim (int): Position embedding dimension fed to experts.
        num_experts (int): Number of sparse experts.
        r (int): Low-rank decomposition rank for each expert.
        num_scales (int): Number of graph aggregation scales.
        top_k (int): Top-k routing for sparse expert selection.
        dropout (float): Dropout probability.
    """

    def __init__(self, hidden_dim: int, pos_dim: int = 8,
                 num_experts: int = 8, r: int = 8,
                 num_scales: int = 3, top_k: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        self.num_experts = num_experts
        self.r = r
        self.num_scales = num_scales
        self.top_k = top_k

        # Multi-scale projection
        self.scale_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_scales)
        ])
        self.scale_fusion = nn.Linear(hidden_dim * num_scales, hidden_dim)

        # Shared transform
        self.shared_transform = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Sparse router with Gumbel-Softmax
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        self.register_buffer('routing_temperature', torch.tensor(1.0))

        # Position-to-expert projection
        self.basis_dim = max(8, min(64, pos_dim * 8))
        self.position_proj = nn.Sequential(
            nn.Linear(pos_dim, self.basis_dim),
            nn.SiLU()
        )

        # Low-rank experts
        self.expert_down = nn.ModuleList([
            nn.Linear(self.basis_dim, r) for _ in range(num_experts)
        ])
        self.expert_up = nn.ModuleList([
            nn.Linear(r, hidden_dim) for _ in range(num_experts)
        ])

        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("expert_counts", torch.zeros(num_experts))
        self.register_buffer("total_tokens", torch.tensor(0.0))
        self.cached_sparse_adj = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _prepare_sparse_adj(self, adj):
        if adj.is_sparse:
            return adj.to_sparse_csr() if adj.layout != torch.sparse_csr else adj
        return adj.detach().to_sparse_csr()

    def _aggregate_multi_scale(self, adj, node_feats):
        """Iterative sparse multi-scale aggregation."""
        B, N, H = node_feats.shape
        outputs = []

        if not hasattr(self, 'cached_sparse_adj') or self.cached_sparse_adj is None:
            sparse_adj = self._prepare_sparse_adj(adj)
        else:
            sparse_adj = self.cached_sparse_adj

        curr_feats = node_feats.transpose(0, 1).reshape(N, B * H)  # (N, B*H)

        for i in range(self.num_scales):
            curr_feats = torch.sparse.mm(sparse_adj, curr_feats)
            agg_k = curr_feats.view(N, B, H).transpose(0, 1)  # (B, N, H)
            agg_k = self.scale_proj[i](agg_k)
            outputs.append(agg_k)

        multi_scale = torch.cat(outputs, dim=-1)
        return self.scale_fusion(multi_scale)

    def forward(self, node_feats: torch.Tensor, adj: torch.Tensor,
                position_embedding: torch.Tensor,
                use_adj_for_pos: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_feats:        (B, N, H)
            adj:               (N, N) adjacency matrix
            position_embedding:(pos_dim, N) position features
            use_adj_for_pos:   whether to propagate position embedding via adj

        Returns:
            output:            (B, N, H)
            load_balance_loss: scalar tensor
        """
        B, N, H = node_feats.shape
        device = node_feats.device

        agg = self._aggregate_multi_scale(adj, node_feats)

        position_embedding = torch.matmul(adj, position_embedding.transpose(0, 1))

        transformed = self.shared_transform(agg)

        # Sparse routing
        router_logits = self.router(transformed.mean(dim=0))  # (N, E)

        if self.training:
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(router_logits) + 1e-8) + 1e-8)
            router_logits = router_logits + gumbel_noise

        router_logits = router_logits / self.routing_temperature.clamp(min=0.1, max=10.0)

        if self.top_k > 0:
            topk_vals, topk_idx = torch.topk(router_logits, self.top_k, dim=-1)
            mask = torch.zeros_like(router_logits).scatter_(1, topk_idx, 1.0)
            router_logits = router_logits * mask + (1 - mask) * (-1e9)

        router_probs = F.softmax(router_logits, dim=-1)  # (N, E)

        # Position-based expert computation
        pos_basis = self.position_proj(position_embedding)
        if pos_basis.dim() == 3:
            pos_basis = pos_basis.mean(dim=0)

        expert_outputs = []
        for down, up in zip(self.expert_down, self.expert_up):
            expert_out = up(F.silu(down(pos_basis)))  # (N, H)
            expert_outputs.append(expert_out)

        expert_outputs = torch.stack(expert_outputs, dim=1)  # (N, E, H)
        expert_outputs = torch.tanh(expert_outputs)

        mixed_output = torch.einsum('ne,neh->nh', router_probs, expert_outputs)
        output = transformed * mixed_output.unsqueeze(0)
        output = self.dropout(self.layernorm(output))

        # Load balance loss
        if self.training:
            self.expert_counts += router_probs.sum(dim=0).detach()
            self.total_tokens += N
            self.routing_temperature *= 0.9999

        avg_expert_prob = router_probs.mean(dim=0)
        target_prob = torch.ones_like(avg_expert_prob) / self.num_experts
        load_balance_loss = F.mse_loss(avg_expert_prob, target_prob)

        return output, load_balance_loss
