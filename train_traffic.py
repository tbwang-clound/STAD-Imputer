"""
Training script for STAD-Imputer on traffic / air-quality benchmarks.

Supported datasets:
  --dataset metrla    METR-LA  (207 sensors, speed, km/h)
  --dataset pemsbay   PEMS-BAY (325 sensors, speed, mph)
  --dataset aqi36     AQI-36   (36 stations, PM2.5)

Usage example:
    conda activate stimp
    python train_traffic.py \
        --dataset metrla \
        --data_root /path/to/PriSTI-main/data \
        --epochs 200 \
        --batch_size 16

The data_root should contain subdirectories:
  metr_la/    pems_bay/    pm25/
matching the layout expected by PriSTI-main.
"""

import os
import sys
import time
import logging
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from timm.utils import AverageMeter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.traffic_dataset import get_dataloader, get_adj, DATASET_NODES
from model.stad_imputer import STAD_Imputer
from utils import check_dir, seed_everything, get_model_size_info, masked_mae, masked_mse

os.environ["CUDA_VISIBLE_DEVICES"] = "6"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(v):
    if isinstance(v, list):
        return v
    try:
        return [int(x) for x in v.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "List elements must be integers separated by commas.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description='STAD-Imputer Traffic Training')

    # ---- Dataset ----
    p.add_argument('--dataset', type=str, default='metrla',
                   choices=['metrla', 'pemsbay', 'aqi36'],
                   help='Which traffic benchmark to use')
    p.add_argument('--data_root', type=str,
                   default='/remote-home/share/dmb_nas/wangtengbo/PriSTI-main/data',
                   help='Root directory that holds metr_la/, pems_bay/, pm25/')
    p.add_argument('--eval_length', type=int, default=24,
                   help='Temporal window length per sample (36 for AQI-36)')
    p.add_argument('--missing_pattern', type=str, default='block',
                   choices=['block', 'point'],
                   help='Missing pattern used for training mask generation')
    p.add_argument('--missing_ratio', type=float, default=0.1,
                   help='Fraction of observed values to mask per training step')

    # ---- Experiment ----
    p.add_argument('--task_name', type=str, default='stad-traffic',
                   help='Tag for checkpoint/log naming')

    # ---- Training ----
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--test_freq', type=int, default=50,
                   help='Run test evaluation every N epochs')
    p.add_argument('--num_workers', type=int, default=4)

    # ---- Diffusion ----
    p.add_argument('--beta_start', type=float, default=0.0001)
    p.add_argument('--beta_end', type=float, default=0.2)
    p.add_argument('--num_steps', type=int, default=50)
    p.add_argument('--num_samples', type=int, default=10,
                   help='Monte-Carlo samples at test time')
    p.add_argument('--schedule', type=str, default='quad',
                   choices=['quad', 'linear'])

    # ---- Shared hidden dimensions ----
    p.add_argument('--hidden_channels', type=int, default=32)
    p.add_argument('--diffusion_embedding_size', type=int, default=64)
    p.add_argument('--balance_weight', type=float, default=0.01)

    # ---- ATI ----
    p.add_argument('--ATI_dim', type=int, default=32)
    p.add_argument('--ATI_dropout', type=float, default=0.1)
    p.add_argument('--ATI_dilation_choices', type=str2list, default=[1, 2, 4, 8])
    p.add_argument('--ATI_tcn_layers', type=int, default=2)

    # ---- ANA ----
    p.add_argument('--ANA_in_dim', type=int, default=32)
    p.add_argument('--ANA_out_dim', type=int, default=32)
    p.add_argument('--ANA_k_phys', type=int, default=8)
    p.add_argument('--ANA_k_feat', type=int, default=8)
    p.add_argument('--ANA_num_prototypes', type=int, default=32)
    p.add_argument('--ANA_dropout', type=float, default=0.1)
    p.add_argument('--Add_ANA_Residual', type=str2bool, default=True)

    # ---- AHM ----
    p.add_argument('--AHM_hidden_dim', type=int, default=32)
    p.add_argument('--AHM_pos_dim', type=int, default=8)
    p.add_argument('--AHM_num_experts', type=int, default=8)
    p.add_argument('--AHM_r', type=int, default=8)
    p.add_argument('--AHM_top_k', type=int, default=3)
    p.add_argument('--AHM_dropout', type=float, default=0.1)
    p.add_argument('--AHM_num_scales', type=int, default=3)

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    config = parser.parse_args()

    # Inject traffic-mode flags
    config.num_nodes = DATASET_NODES[config.dataset]
    # Sentinel values so STADBackbone knows it's not a coastal run
    config.data_root = config.data_root   # kept for reference
    config.area = None

    # Eval_length override for AQI-36
    if config.dataset == 'aqi36' and config.eval_length == 24:
        config.eval_length = 36

    base_dir = os.path.join(
        "checkpoints",
        f"{config.dataset}_{config.task_name}"
    )
    check_dir(base_dir)
    seed_everything(1234)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(base_dir, f"{timestamp}_{config.dataset}.log")
    logging.basicConfig(
        level=logging.INFO,
        filename=log_file, filemode='a',
        format='%(asctime)s - %(message)s'
    )
    print(config)
    logging.info(config)

    device = torch.device("cuda:0") if torch.cuda.is_available() \
        else torch.device("cpu")

    # ---- Data ----
    train_loader, valid_loader, test_loader, train_mean, train_std = \
        get_dataloader(
            dataset_name=config.dataset,
            data_root=config.data_root,
            batch_size=config.batch_size,
            eval_length=config.eval_length,
            missing_pattern=config.missing_pattern,
            missing_ratio=config.missing_ratio,
            num_workers=config.num_workers,
        )

    # ---- Adjacency matrix ----
    adj_np = get_adj(config.dataset, config.data_root)
    adj = torch.from_numpy(adj_np).float().to(device)

    N = config.num_nodes

    # ---- Bounds (use train statistics to set normalised range) ----
    # For traffic: data are z-normalised, so bounds are set symmetrically.
    # We clip at ±5 sigma in normalised space to avoid extreme outliers.
    low_bound = torch.full((N,), -5.0).to(device)
    high_bound = torch.full((N,), 5.0).to(device)

    # ---- Model ----
    model = STAD_Imputer(config, low_bound, high_bound).to(device)
    get_model_size_info(model)

    # ---- Optimizer & scheduler ----
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.wd)
    p1 = int(0.75 * config.epochs)
    p2 = int(0.90 * config.epochs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1)

    # Inverse-normalisation tensors
    t_mean = torch.from_numpy(train_mean).float().to(device)
    t_std = torch.from_numpy(train_std).float().to(device)

    best_real_mae = 1e9

    # =========================================================================
    # Training loop
    # =========================================================================
    pbar = tqdm(range(1, config.epochs + 1))
    for epoch in pbar:
        epoch_start = time.time()
        losses_m = AverageMeter()
        model.train()
        scheduler.step()

        for batch in train_loader:
            # batch: (ob_data, ob_mask, gt_mask, cond_mask)  shapes (B, T, 1, N)
            ob_data, ob_mask, gt_mask, cond_mask = batch
            ob_data = ob_data.float().to(device)   # (B, T, 1, N)
            ob_mask = ob_mask.float().to(device)
            # trainstep expects (B, T, K, N), K=1 → already correct
            loss = model.trainstep(ob_data, ob_mask, adj, is_train=1)
            losses_m.update(loss.item(), ob_data.size(0))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        epoch_dur = time.time() - epoch_start
        log_str = (f"Epoch {epoch:04d} | Loss: {losses_m.avg:.4f} | "
                   f"Elapsed: {epoch_dur:.2f}s")
        pbar.set_description(log_str)
        pbar.write(log_str)

        # =====================================================================
        # Evaluation
        # =====================================================================
        if epoch % config.test_freq == 0 and epoch != 0:
            norm_mae_list, norm_mse_list = [], []
            real_mae_list, real_mse_list = [], []

            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    ob_data, ob_mask, gt_mask, cond_mask = batch
                    ob_data = ob_data.float().to(device)   # (B, T, 1, N)
                    ob_mask = ob_mask.float().to(device)
                    gt_mask = gt_mask.float().to(device)

                    # impute: observed_mask = gt_mask (the "seen" portion)
                    samples = model.impute(
                        ob_data, gt_mask, adj, config.num_samples)
                    # samples: (B, n_samples, T, 1, N)

                    imputed = samples.median(dim=1).values  # (B, T, 1, N)

                    # Squeeze K dim
                    pred_norm = imputed[:, :, 0, :]    # (B, T, N)
                    true_norm = ob_data[:, :, 0, :]
                    ob_mask_2d = ob_mask[:, :, 0, :]
                    gt_mask_2d = gt_mask[:, :, 0, :]

                    # Evaluation mask: pixels that are observed but hidden from
                    # the model (ob_mask=1 and gt_mask=0)
                    mask_eval = (ob_mask_2d - gt_mask_2d).clamp(0, 1)

                    p_n = pred_norm.cpu()
                    t_n = true_norm.cpu()
                    m_e = mask_eval.cpu()

                    norm_mae_list.append(masked_mae(p_n, t_n, m_e).item())
                    norm_mse_list.append(masked_mse(p_n, t_n, m_e).item())

                    # Inverse normalise to physical scale
                    # t_mean / t_std have shape (N,), broadcast over (B, T, N)
                    # Use cpu copies to avoid device mismatch (pred_norm already on cpu)
                    _std = t_std.cpu().unsqueeze(0).unsqueeze(0)
                    _mean = t_mean.cpu().unsqueeze(0).unsqueeze(0)
                    pred_real = p_n * _std + _mean
                    true_real = t_n * _std + _mean

                    p_r = pred_real.cpu()
                    t_r = true_real.cpu()
                    real_mae_list.append(masked_mae(p_r, t_r, m_e).item())
                    real_mse_list.append(masked_mse(p_r, t_r, m_e).item())

            def safe_mean(lst):
                vals = [x for x in lst if x == x]  # drop NaN
                return float(np.mean(vals)) if vals else 0.0

            final_norm_mae = safe_mean(norm_mae_list)
            final_norm_rmse = float(np.sqrt(safe_mean(norm_mse_list)))
            final_real_mae = safe_mean(real_mae_list)
            final_real_rmse = float(np.sqrt(safe_mean(real_mse_list)))

            eval_str = (
                f"Eval @ Epoch {epoch}:\n"
                f"  [Norm]  MAE={final_norm_mae:.4f}  RMSE={final_norm_rmse:.4f}\n"
                f"  [Real]  MAE={final_real_mae:.4f}  RMSE={final_real_rmse:.4f}"
            )
            print(eval_str)
            logging.info(eval_str)

            if final_real_mae < best_real_mae:
                best_real_mae = final_real_mae
                save_path = os.path.join(
                    base_dir,
                    f"best_{config.dataset}_{config.task_name}.pt"
                )
                torch.save(model, save_path)
                msg = (f"*** Best model saved "
                       f"(Real MAE={best_real_mae:.4f}) -> {save_path}")
                print(msg)
                logging.info(msg)


if __name__ == '__main__':
    main()
