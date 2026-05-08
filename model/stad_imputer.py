"""
STAD-Imputer: Spatio-Temporal Adaptive Diffusion Imputer

Main model implementation integrating three adaptive modules:
  - ATI: Adaptive Temporal Integrator
  - ANA: Adaptive Neighborhood Aggregator
  - AHM: Adaptive Heterogeneity Modulator
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.modules import ATI, ANA, AHM


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters | Total: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M")
    return total, trainable


# ---------------------------------------------------------------------------
# Diffusion Embedding
# ---------------------------------------------------------------------------

class DiffusionEmbedding(nn.Module):
    """Sinusoidal diffusion step embedding."""

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
        steps = torch.arange(num_steps).unsqueeze(1)          # (T, 1)
        freqs = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * freqs
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)


# ---------------------------------------------------------------------------
# GCN (fallback spatial aggregator; STAD-Imputer uses AHM by default)
# ---------------------------------------------------------------------------

class GCN(nn.Module):
    """Heterogeneous GCN used as the baseline spatial encoder."""

    def __init__(self, c_in, c_out, c_hid, num_types):
        super().__init__()
        self.linear = nn.Linear(c_in, c_out, bias=False)
        self.num_types = num_types
        nn.init.uniform_(self.linear.weight,
                         -np.sqrt(6 / (c_in + c_out)),
                         np.sqrt(6 / (c_in + c_out)))
        self.weights_pool = nn.init.xavier_normal_(
            nn.Parameter(torch.FloatTensor(c_hid, c_in, c_out)))

    def forward(self, node_feats, adj_matrix, position_embedding):
        node_feats = torch.matmul(adj_matrix, node_feats)
        pos = torch.matmul(adj_matrix, position_embedding.transpose(0, 1))
        pos_weights = torch.einsum('nd, dio-> nio', pos, self.weights_pool)
        return torch.einsum('bni, nio->bno', node_feats, pos_weights)


# ---------------------------------------------------------------------------
# Core denoising backbone
# ---------------------------------------------------------------------------

class STADBackbone(nn.Module):
    """
    Spatio-temporal denoising backbone that integrates ATI, ANA, and AHM.

    Input tensor shape convention:
        x : (B, T, K, 2, N)  where K=1, dim-3 has [observed, noisy]
    """

    def __init__(self, config, low_bound, high_bound):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda:0") if torch.cuda.is_available() \
            else torch.device("cpu")

        # ---- Detect mode: coastal (spatial grid) vs traffic (sensor graph) ----
        # Traffic mode is activated when config has attribute `num_nodes`
        self.traffic_mode = getattr(config, 'num_nodes', None) is not None

        # I/O projections
        self.input_projection = Conv1d_with_init(2, config.hidden_channels, 1)
        self.mid_projection = Conv1d_with_init(
            config.hidden_channels, 2 * config.hidden_channels, 1)
        self.output_projection = Conv1d_with_init(config.hidden_channels, 1, 1)

        if not self.traffic_mode:
            # ---- Coastal: load geographic auxiliary files ----
            self.is_sea = torch.from_numpy(
                np.load(f"{config.data_root}/{config.area}/is_sea.npy")
            ).to(self.device).bool()
            self.mean = torch.from_numpy(
                np.load(f"{config.data_root}/{config.area}/mean.npy")
            ).to(self.device)
            self.std = torch.from_numpy(
                np.load(f"{config.data_root}/{config.area}/std.npy")
            ).to(self.device)
            # Position embedding: [mean, std, pos_w, pos_h] -> project to hidden//4
            self.projection1 = nn.Linear(4, config.hidden_channels // 4)
            learnable_pe = self._build_position_embedding()[:, self.is_sea]
            self.register_buffer("embedding", learnable_pe.float())
        else:
            # ---- Traffic: learnable node embedding (no geographic files) ----
            N = config.num_nodes
            # Learnable 2-dim position encoding for each node
            self.node_emb = nn.Embedding(N, config.hidden_channels // 4)
            nn.init.xavier_uniform_(self.node_emb.weight)

        # Diffusion embedding
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config.num_steps,
            embedding_dim=config.diffusion_embedding_size,
            projection_dim=config.hidden_channels
        )

        self.norm1 = nn.LayerNorm(config.hidden_channels)
        self.norm2 = nn.LayerNorm(config.hidden_channels)

        # ATI: Adaptive Temporal Integrator
        self.ati = ATI(
            input_dim=config.hidden_channels,
            hidden_dim=config.ATI_dim,
            tcn_layers=config.ATI_tcn_layers,
            mode='imputation',
            dropout=config.ATI_dropout,
            dilation_choices=config.ATI_dilation_choices,
        )

        # ANA: Adaptive Neighborhood Aggregator
        self.ana = ANA(
            in_dim=config.ANA_in_dim,
            out_dim=config.ANA_out_dim,
            k_phys=config.ANA_k_phys,
            k_feat=config.ANA_k_feat,
            num_prototypes=config.ANA_num_prototypes,
            dropout=config.ANA_dropout,
        )

        # AHM: Adaptive Heterogeneity Modulator
        self.ahm = AHM(
            hidden_dim=config.AHM_hidden_dim,
            pos_dim=config.AHM_pos_dim,
            num_experts=config.AHM_num_experts,
            r=config.AHM_r,
            top_k=config.AHM_top_k,
            dropout=config.AHM_dropout,
            num_scales=config.AHM_num_scales,
        )

        self.gn = nn.GroupNorm(4, config.hidden_channels)

        self.low_bound = low_bound
        self.high_bound = high_bound

        # Flag to print parameter counts once
        self._printed_params = False

    # ------------------------------------------------------------------
    # Position embedding helpers
    # ------------------------------------------------------------------

    def _build_position_embedding(self):
        height = self.config.height
        width = self.config.width
        pos_w = torch.arange(0., width) / width
        pos_h = torch.arange(0., height) / height
        pos_w = pos_w.unsqueeze(0).expand(height, -1)
        pos_h = pos_h.unsqueeze(1).expand(-1, width)
        pe = torch.stack([self.mean.cpu(), self.std.cpu(), pos_w, pos_h], dim=0)
        return pe.to(self.device)

    def _get_position_embedding(self):
        if self.traffic_mode:
            # Traffic mode: learnable node embedding (hidden//4, N)
            N = self.config.num_nodes
            idx = torch.arange(N, device=self.node_emb.weight.device)
            emb = self.node_emb(idx)          # (N, hidden//4)
            return emb.transpose(0, 1)        # (hidden//4, N)
        # Coastal mode
        x = self.embedding.transpose(0, 1)
        x = self.projection1(x)
        return x.transpose(0, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, mask, adj, diffusion_step):
        """
        Args:
            x:             (B, T, K, 2, N)
            mask:          (B, T, K, N)
            adj:           (N, N)
            diffusion_step:(B,)
        Returns:
            y:              (B, T, K, N)
            load_bal_loss:  scalar
            ortho_loss:     scalar (always 0, kept for interface compatibility)
        """
        B, T, K, C, N = x.shape

        # ---- Input projection ----
        x = rearrange(x, 'b t k c n -> (b t k) c n')
        x = self.input_projection(x)
        x = rearrange(x, '(b t k) c n -> b t k c n', b=B, t=T, k=K)

        # ---- Diffusion step embedding ----
        diff_emb = self.diffusion_embedding(diffusion_step)
        diff_emb = diff_emb.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        x = x + diff_emb

        # ---- ATI: temporal encoding ----
        if not self._printed_params:
            count_parameters(self.ati)
            count_parameters(self.ahm)
            self._printed_params = True

        x = rearrange(x, 'b t k c n -> (b k n) t c')
        x = self.ati(x, B, N)                    # (B*K*N, T, C)
        x = rearrange(x, '(b k n) t c -> (b k t) n c', b=B, k=K, n=N)

        x_skip = x  # save for skip connection

        # ---- Position embedding ----
        pos_emb = self._get_position_embedding()  # (C//4, N)

        # ---- ANA: neighborhood aggregation ----
        x = self.ana(x, adj)                      # (B*K*T, N, C)

        if self.config.Add_ANA_Residual:
            x = x + x_skip
            x_skip = x

        # ---- AHM: heterogeneity modulation ----
        x, load_bal_loss = self.ahm(x, adj, pos_emb)

        # ---- Output ----
        x = rearrange(x, '(b k t) n c -> (b t k) c n', b=B, k=K, t=T)
        x_skip = rearrange(x_skip, '(b k t) n c -> (b t k) c n', b=B, k=K, t=T)

        x = x + x_skip
        x = self.gn(x)
        x = self.mid_projection(x)
        gate, filt = torch.chunk(x, 2, dim=1)
        x = torch.sigmoid(gate) * torch.tanh(filt)
        y = self.output_projection(x)
        y = rearrange(y, '(b t k) c n -> b t k c n', b=B, k=K, t=T)
        y = y.squeeze(3)

        # Clamp to valid physical range
        lo = self.low_bound.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(y)
        hi = self.high_bound.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(y)
        y = torch.clamp(y, lo, hi)

        return y, load_bal_loss, 0


# ---------------------------------------------------------------------------
# STAD_Imputer: top-level diffusion wrapper
# ---------------------------------------------------------------------------

class STAD_Imputer(nn.Module):
    """
    Spatio-Temporal Adaptive Diffusion Imputer (STAD-Imputer).

    A unified conditional diffusion framework for highly sparse coastal
    remote sensing imputation.

    Args:
        config:     Namespace / argparse config with all hyperparameters.
        low_bound:  (N,) tensor, per-node lower bound in normalized space.
        high_bound: (N,) tensor, per-node upper bound in normalized space.
    """

    def __init__(self, config, low_bound, high_bound):
        super().__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() \
            else torch.device("cpu")
        self.config = config
        self.num_steps = config.num_steps

        self.backbone = STADBackbone(config, low_bound, high_bound)

        # Noise schedule
        if config.schedule == "quad":
            self.beta = torch.linspace(
                config.beta_start ** 0.5, config.beta_end ** 0.5,
                self.num_steps
            ) ** 2
        else:  # linear
            self.beta = torch.linspace(
                config.beta_start, config.beta_end, self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = torch.cumprod(self.alpha_hat, dim=0)
        self.alpha_prev = F.pad(self.alpha[:-1], (1, 0), value=1.)
        self.alpha_torch = (
            self.alpha.float().to(self.device)
            .unsqueeze(1).unsqueeze(1).unsqueeze(1)
        )

        self.low_bound = low_bound
        self.high_bound = high_bound

    # ------------------------------------------------------------------
    # Random masking for training
    # ------------------------------------------------------------------

    def get_randmask(self, observed_mask, sample_ratio):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def trainstep(self, observed_data, observed_mask, adj, is_train=1, set_t=-1):
        """
        One training step: add noise and compute denoising loss.

        Args:
            observed_data:  (B, T, K, N) normalized observations
            observed_mask:  (B, T, K, N) binary mask (1 = observed)
            adj:            (N, N) adjacency matrix
            is_train (int): 1 for training, 0 for validation
            set_t (int):    fixed diffusion step for validation

        Returns:
            loss: scalar training loss
        """
        cond_mask = self.get_randmask(observed_mask,
                                      self.config.missing_ratio).to(self.device)
        B = observed_data.shape[0]

        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.config.num_steps, [B]).to(self.device)

        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(observed_data)

        # Mean fill for unobserved positions
        mean = (observed_data * cond_mask).sum(dim=1, keepdim=True) / \
               (cond_mask.sum(dim=1, keepdim=True) + 1e-5)
        observed_data_imputed = torch.where(cond_mask.bool(), observed_data,
                                            mean.expand_as(observed_data))

        noisy_data = (current_alpha ** 0.5) * observed_data_imputed + \
                     (1.0 - current_alpha) ** 0.5 * noise

        total_input = torch.stack(
            [observed_data_imputed, (1 - cond_mask) * noisy_data], dim=3
        )

        predicted, load_balance_loss, _ = self.backbone(
            total_input, cond_mask, adj, t)

        target_mask = observed_mask - cond_mask
        residual = (observed_data - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

        return loss + self.config.balance_weight * load_balance_loss

    # ------------------------------------------------------------------
    # Imputation (inference)
    # ------------------------------------------------------------------

    def impute(self, observed_data, observed_mask, adj, n_samples):
        """
        DDPM reverse diffusion to generate n imputation samples.

        Args:
            observed_data:  (B, T, K, N)
            observed_mask:  (B, T, K, N)
            adj:            (N, N)
            n_samples (int): number of Monte-Carlo samples

        Returns:
            imputed_samples: (B, n_samples, T, K, N)
        """
        B, T, K, N = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, T, K, N)

        mean = (observed_data * observed_mask).sum(1, keepdim=True) / \
               (observed_mask.sum(1, keepdim=True) + 1e-5)
        observed_data_imputed = torch.where(
            observed_mask.bool(), observed_data, mean.expand_as(observed_data))

        with torch.no_grad():
            for i in range(n_samples):
                current_sample = torch.randn_like(observed_data).to(self.device) + \
                                 mean.expand_as(observed_data)

                for t in range(self.num_steps - 1, -1, -1):
                    total_input = torch.stack(
                        [observed_data_imputed,
                         (1 - observed_mask) * current_sample], dim=3
                    )
                    predicted, _, _ = self.backbone(
                        total_input, observed_mask, adj,
                        (torch.ones(B) * t).long().to(self.device)
                    )

                    coeff1 = (1 - self.alpha_prev[t]) * (self.alpha_hat[t]) ** 0.5 / \
                             (1 - self.alpha[t])
                    coeff2 = ((1 - self.alpha_hat[t]) * (self.alpha_prev[t]) ** 0.5) / \
                             (1 - self.alpha[t])
                    current_sample = coeff1 * current_sample + coeff2 * predicted

                    if t > 0:
                        noise = torch.randn_like(current_sample)
                        sigma = (
                            (1.0 - self.alpha[t - 1]) /
                            (1.0 - self.alpha[t]) * self.beta[t]
                        ) ** 0.5
                        current_sample += sigma * noise

                imputed_samples[:, i] = current_sample.detach().cpu()

        return imputed_samples
