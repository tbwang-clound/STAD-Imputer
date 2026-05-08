"""
Training script for STAD-Imputer.

Usage example:
    python train.py \
        --data_root /path/to/data \
        --area PRE \
        --datasets_type sst4
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

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.dataset import CoastalImputationDataset
from model.stad_imputer import STAD_Imputer
from utils import (
    check_dir, seed_everything, get_model_size_info,
    masked_mae, masked_mse, masked_mape,
    masked_r2, masked_ssim, calculate_crps,
)


# ---------------------------------------------------------------------------
# Argument helpers
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
            "List elements must be integers separated by commas (e.g. 1,2,4,8)"
        )

import os
#import uuid
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description='STAD-Imputer Training')

    # ---- Experiment identity ----
    parser.add_argument('--task_name', type=str,
                        default='stad-imputer-default',
                        help='Experiment tag used for log/checkpoint naming')

    # ---- Data ----
    parser.add_argument('--data_root', type=str,
                        default='/remote-home/share/dmb_nas/wangtengbo/zone_sst4_data',
                        help='Root directory containing zone data folders')
    parser.add_argument('--area', type=str, default='PRE',
                        help='Spatial region (PRE / Yangtze / Chesapeake / MEXICO)')
    parser.add_argument('--datasets_type', type=str, default='sst4',
                        help='Variable type: chla / par / sst4 / sst11')
    parser.add_argument('--datasets_standard_type', type=str, default='repete',
                        help='Normalization strategy (repete)')

    # ---- Task ----
    parser.add_argument('--in_len', type=int, default=46,
                        help='Input sequence length (time steps)')
    parser.add_argument('--out_len', type=int, default=0,
                        help='Prediction horizon (0 = pure imputation)')
    parser.add_argument('--missing_ratio', type=float, default=0.9,
                        help='Simulated missing rate during training')

    # ---- Training ----
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--test_freq', type=int, default=500,
                        help='Run test evaluation every N epochs')

    # ---- Diffusion ----
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.2)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of Monte-Carlo imputation samples at test time')
    parser.add_argument('--schedule', type=str, default='quad',
                        choices=['quad', 'linear'])

    # ---- Shared hidden dimensions ----
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--diffusion_embedding_size', type=int, default=64)
    parser.add_argument('--balance_weight', type=float, default=0.01,
                        help='Weight for MoE load-balance loss')

    # ---- ATI hyperparameters ----
    parser.add_argument('--ATI_dim', type=int, default=32)
    parser.add_argument('--ATI_dropout', type=float, default=0.1)
    parser.add_argument('--ATI_dilation_choices', type=str2list,
                        default=[1, 2, 4, 8])
    parser.add_argument('--ATI_tcn_layers', type=int, default=2)

    # ---- ANA hyperparameters ----
    parser.add_argument('--ANA_in_dim', type=int, default=32)
    parser.add_argument('--ANA_out_dim', type=int, default=32)
    parser.add_argument('--ANA_k_phys', type=int, default=8)
    parser.add_argument('--ANA_k_feat', type=int, default=8)
    parser.add_argument('--ANA_num_prototypes', type=int, default=32)
    parser.add_argument('--ANA_dropout', type=float, default=0.1)
    parser.add_argument('--Add_ANA_Residual', type=str2bool, default=True)

    # ---- AHM hyperparameters ----
    parser.add_argument('--AHM_hidden_dim', type=int, default=32)
    parser.add_argument('--AHM_pos_dim', type=int, default=8)
    parser.add_argument('--AHM_num_experts', type=int, default=8)
    parser.add_argument('--AHM_r', type=int, default=8)
    parser.add_argument('--AHM_top_k', type=int, default=3)
    parser.add_argument('--AHM_dropout', type=float, default=0.1)
    parser.add_argument('--AHM_num_scales', type=int, default=3)

    # ---- Misc ----
    parser.add_argument('--mixer_position_embedding', type=str2bool,
                        default=False, help='Use absolute sinusoidal position encoding')
    parser.add_argument('--Entire_Relative_Position_Embedding', type=str2bool,
                        default=True)

    return parser


# ---------------------------------------------------------------------------
# Area config
# ---------------------------------------------------------------------------

AREA_SHAPES = {
    'MEXICO':     (36, 120),
    'PRE':        (60, 96),
    'Chesapeake': (60, 48),
    'Yangtze':    (96, 72),
}


def set_area_config(config):
    if config.area in AREA_SHAPES:
        config.height, config.width = AREA_SHAPES[config.area]
    else:
        raise ValueError(f"Unknown area: {config.area}. "
                         f"Supported: {list(AREA_SHAPES.keys())}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    config = parser.parse_args()
    set_area_config(config)

    # Output directory
    base_dir = os.path.join(
        "checkpoints",
        f"{config.in_len}_{config.area}_{config.datasets_type}_{config.task_name}"
    )
    check_dir(base_dir)
    seed_everything(1234)

    # Logging
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(
        base_dir,
        f"{timestamp}_miss{config.missing_ratio}_{config.datasets_type}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        filename=log_file, filemode='a',
        format='%(asctime)s - %(message)s'
    )
    print(config)
    logging.info(config)

    device = torch.device("cuda:0") if torch.cuda.is_available() \
        else torch.device("cpu")

    # ---- Datasets ----
    train_dataset = CoastalImputationDataset(config, mode='train')
    train_loader = DataLoader(train_dataset, config.batch_size,
                              shuffle=True, prefetch_factor=2, num_workers=2)

    test_dataset = CoastalImputationDataset(config, mode='test')
    test_loader = DataLoader(test_dataset, config.batch_size,
                             shuffle=False, num_workers=2)

    # ---- Adjacency matrix ----
    adj = np.load(os.path.join(config.data_root, config.area, 'adj.npy'))
    adj = torch.from_numpy(adj).float().to(device)

    # ---- Bounds for clamping model output ----
    low_bound = torch.from_numpy(train_dataset.min).float().to(device)
    high_bound = torch.from_numpy(train_dataset.max).float().to(device)

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

    # ---- Scaler for inverse normalization ----
    scaler_mean = scaler_std = None
    if config.datasets_type != 'chla' and test_dataset.mean is not None:
        scaler_mean = torch.from_numpy(test_dataset.mean).float().to(device)
        scaler_std = torch.from_numpy(test_dataset.std).float().to(device)

    best_real_mae = 1e9

    # =========================================================================
    # Training loop
    # =========================================================================
    pbar = tqdm(range(1, config.epochs + 1))
    for epoch in pbar:
        epoch_start = time.time()
        losses_m = AverageMeter()
        data_time_m = AverageMeter()
        model.train()
        scheduler.step()
        end = time.time()

        for (datas, data_ob_masks, data_gt_masks, labels, label_masks) \
                in train_loader:
            datas = datas.float().to(device)
            data_ob_masks = data_ob_masks.to(device)
            data_gt_masks = data_gt_masks.to(device)
            labels = labels.to(device)
            label_masks = label_masks.to(device)

            loss = model.trainstep(datas, data_ob_masks, adj, is_train=1)
            losses_m.update(loss.item(), datas.size(0))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            torch.cuda.synchronize()
            data_time_m.update(time.time() - end)
            end = time.time()

        epoch_dur = time.time() - epoch_start
        log_str = (f"Epoch {epoch:04d} | Loss: {losses_m.avg:.4f} | "
                   f"Batch: {data_time_m.avg:.3f}s | Total: {epoch_dur:.2f}s")
        pbar.set_description(log_str)
        pbar.write(log_str)

        # =====================================================================
        # Evaluation
        # =====================================================================
        if epoch % config.test_freq == 0 and epoch != 0:
            norm_mae_list, norm_mse_list = [], []
            real_mae_list, real_mse_list = [], []
            real_mape_list, real_r2_list, real_ssim_list, real_crps_list = [], [], [], []

            model.eval()
            with torch.no_grad():
                for (datas, data_ob_masks, data_gt_masks,
                     labels, label_masks) in test_loader:
                    datas = datas.float().to(device)
                    data_ob_masks = data_ob_masks.to(device)
                    data_gt_masks = data_gt_masks.to(device)

                    samples = model.impute(
                        datas, data_gt_masks, adj, config.num_samples)
                    imputed = samples.median(dim=1).values  # (B, T, K, N)

                    # Squeeze channel dim K=1
                    if imputed.dim() == 4:
                        pred_norm = imputed[:, :, 0, :]
                        true_norm = datas[:, :, 0, :]
                        samples_norm = samples[:, :, :, 0, :]
                    else:
                        pred_norm = imputed
                        true_norm = datas
                        samples_norm = samples

                    mask_eval = (data_ob_masks - data_gt_masks)
                    if mask_eval.dim() == 4:
                        mask_eval = mask_eval[:, :, 0, :]

                    # Normalized metrics
                    p_n = pred_norm.cpu()
                    t_n = true_norm.cpu()
                    m_n = mask_eval.cpu()
                    norm_mae_list.append(masked_mae(p_n, t_n, m_n).item())
                    norm_mse_list.append(masked_mse(p_n, t_n, m_n).item())

                    # Inverse normalize
                    if config.datasets_type == 'chla':
                        pred_real = torch.pow(10, pred_norm)
                        true_real = torch.pow(10, true_norm)
                        samp_real = torch.pow(10, samples_norm)
                    elif scaler_mean is not None:
                        pred_norm = pred_norm.to(device)
                        samples_norm = samples_norm.to(device)
                        pred_real = pred_norm * scaler_std + scaler_mean
                        true_real = true_norm * scaler_std + scaler_mean
                        samp_real = samples_norm * scaler_std + scaler_mean
                    else:
                        pred_real, true_real, samp_real = pred_norm, true_norm, samples_norm

                    p_r = pred_real.cpu()
                    t_r = true_real.cpu()
                    m_r = mask_eval.cpu()
                    s_r = samp_real.cpu()

                    real_mae_list.append(masked_mae(p_r, t_r, m_r).item())
                    real_mse_list.append(masked_mse(p_r, t_r, m_r).item())
                    real_mape_list.append(masked_mape(p_r, t_r, m_r).item())
                    real_r2_list.append(masked_r2(p_r, t_r, m_r).item())
                    real_ssim_list.append(masked_ssim(p_r, t_r, m_r).item())
                    real_crps_list.append(calculate_crps(s_r, t_r, m_r).item())

            # Aggregate
            def safe_mean(lst):
                vals = [x for x in lst if x != 0]
                return float(np.mean(vals)) if vals else 0.0

            final_norm_mae = safe_mean(norm_mae_list)
            final_norm_mse = safe_mean(norm_mse_list)
            final_real_mae = safe_mean(real_mae_list)
            final_real_mse = safe_mean(real_mse_list)
            final_real_rmse = float(np.sqrt(final_real_mse))
            final_real_mape = safe_mean(real_mape_list)
            final_real_r2 = safe_mean(real_r2_list)
            final_real_ssim = safe_mean(real_ssim_list)
            final_real_crps = safe_mean(real_crps_list)

            eval_str = (
                f"Eval @ Epoch {epoch}:\n"
                f"  [Norm]  MAE={final_norm_mae:.4f}  MSE={final_norm_mse:.4f}\n"
                f"  [Real]  MAE={final_real_mae:.4f}  RMSE={final_real_rmse:.4f}"
                f"  MAPE={final_real_mape:.4f}\n"
                f"  [Real]  R2={final_real_r2:.4f}  SSIM={final_real_ssim:.4f}"
                f"  CRPS={final_real_crps:.4f}"
            )
            print(eval_str)
            logging.info(eval_str)

            # Save best model
            if final_real_mae < best_real_mae:
                best_real_mae = final_real_mae
                save_path = os.path.join(
                    base_dir,
                    f"best_{config.missing_ratio}_{config.task_name}"
                    f"_{config.area}_{config.datasets_type}.pt"
                )
                torch.save(model, save_path)
                msg = f"*** Best model saved (Real MAE={best_real_mae:.4f}) -> {save_path}"
                print(msg)
                logging.info(msg)


if __name__ == '__main__':
    main()
