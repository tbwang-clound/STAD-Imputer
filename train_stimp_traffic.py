"""
Training script for STIMP (graphdiffusion) baseline on traffic / air-quality benchmarks.

Model: model/stimp_traffic.py (adapted from STIMP-release/model/graphdiffusion.py)
Data loader: reuses dataset/traffic_dataset.py for consistency with STAD-Imputer

Supported datasets:
  --dataset metrla    METR-LA  (207 sensors, speed, km/h)
  --dataset pemsbay   PEMS-BAY (325 sensors, speed, mph)
  --dataset aqi36     AQI-36   (36 stations, PM2.5)
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
from model.stimp_traffic import IAP_base as STIMP_Base
from utils import check_dir, seed_everything, get_model_size_info, masked_mae, masked_mse


def build_parser():
    p = argparse.ArgumentParser(description='STIMP Traffic Training')

    # ---- Dataset ----
    p.add_argument('--dataset', type=str, default='aqi36',
                   choices=['metrla', 'pemsbay', 'aqi36'])
    p.add_argument('--data_root', type=str,
                   default='/remote-home/share/dmb_nas/wangtengbo/PriSTI-main/data')
    p.add_argument('--eval_length', type=int, default=36)
    p.add_argument('--missing_pattern', type=str, default='point',
                   choices=['block', 'point'])
    p.add_argument('--missing_ratio', type=float, default=0.9)

    # ---- Experiment ----
    p.add_argument('--task_name', type=str, default='stimp-traffic')

    # ---- Training ----
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--test_freq', type=int, default=100)
    p.add_argument('--num_workers', type=int, default=4)

    # ---- Diffusion ----
    p.add_argument('--beta_start', type=float, default=0.0001)
    p.add_argument('--beta_end', type=float, default=0.2)
    p.add_argument('--num_steps', type=int, default=50)
    p.add_argument('--num_samples', type=int, default=10)
    p.add_argument('--schedule', type=str, default='quad', choices=['quad', 'linear'])

    # ---- STIMP hidden dims ----
    p.add_argument('--hidden_channels', type=int, default=32)
    p.add_argument('--diffusion_embedding_size', type=int, default=64)

    return p


def main():
    parser = build_parser()
    config = parser.parse_args()

    config.num_nodes = DATASET_NODES[config.dataset]
    config.area = None

    if config.dataset == 'aqi36' and config.eval_length == 24:
        config.eval_length = 36

    base_dir = os.path.join("checkpoints", f"{config.dataset}_{config.task_name}")
    check_dir(base_dir)
    seed_everything(1234)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(base_dir, f"{timestamp}_{config.dataset}.log")
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a',
                        format='%(asctime)s - %(message)s')
    print(config)
    logging.info(config)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # ---- Data ----
    train_loader, valid_loader, test_loader, train_mean, train_std = \
        get_dataloader(
            dataset_name=config.dataset, data_root=config.data_root,
            batch_size=config.batch_size, eval_length=config.eval_length,
            missing_pattern=config.missing_pattern, missing_ratio=config.missing_ratio,
            num_workers=config.num_workers)

    # ---- Adjacency matrix ----
    adj_np = get_adj(config.dataset, config.data_root)
    adj = torch.from_numpy(adj_np).float().to(device)

    N = config.num_nodes

    # ---- Bounds ----
    low_bound = torch.full((N,), -5.0).to(device)
    high_bound = torch.full((N,), 5.0).to(device)

    # ---- Model ----
    model = STIMP_Base(config, low_bound, high_bound).to(device)
    get_model_size_info(model)

    # ---- Optimizer & scheduler ----
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)
    p1 = int(0.75 * config.epochs)
    p2 = int(0.90 * config.epochs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1)

    t_mean = torch.from_numpy(train_mean).float()
    t_std = torch.from_numpy(train_std).float()

    best_real_mae = 1e9

    pbar = tqdm(range(1, config.epochs + 1))
    for epoch in pbar:
        epoch_start = time.time()
        losses_m = AverageMeter()
        model.train()
        scheduler.step()

        for batch in train_loader:
            ob_data, ob_mask, gt_mask, cond_mask = batch
            ob_data = ob_data.float().to(device)
            ob_mask = ob_mask.float().to(device)

            loss = model.trainstep(ob_data, ob_mask, adj, is_train=1)
            losses_m.update(loss.item(), ob_data.size(0))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        epoch_dur = time.time() - epoch_start
        log_str = f"Epoch {epoch:04d} | Loss: {losses_m.avg:.4f} | Elapsed: {epoch_dur:.2f}s"
        pbar.set_description(log_str)
        pbar.write(log_str)

        # ---- Evaluation ----
        if epoch % config.test_freq == 0 and epoch != 0:
            norm_mae_list, norm_mse_list = [], []
            real_mae_list, real_mse_list = [], []

            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    ob_data, ob_mask, gt_mask, cond_mask = batch
                    ob_data = ob_data.float().to(device)
                    ob_mask = ob_mask.float().to(device)
                    gt_mask = gt_mask.float().to(device)

                    samples = model.impute(ob_data, gt_mask, adj, config.num_samples)
                    imputed = samples.median(dim=1).values

                    pred_norm = imputed[:, :, 0, :]
                    true_norm = ob_data[:, :, 0, :]
                    ob_mask_2d = ob_mask[:, :, 0, :]
                    gt_mask_2d = gt_mask[:, :, 0, :]
                    mask_eval = (ob_mask_2d - gt_mask_2d).clamp(0, 1)

                    p_n = pred_norm.cpu()
                    t_n = true_norm.cpu()
                    m_e = mask_eval.cpu()

                    norm_mae_list.append(masked_mae(p_n, t_n, m_e).item())
                    norm_mse_list.append(masked_mse(p_n, t_n, m_e).item())

                    _std = t_std.unsqueeze(0).unsqueeze(0)
                    _mean = t_mean.unsqueeze(0).unsqueeze(0)
                    pred_real = p_n * _std + _mean
                    true_real = t_n * _std + _mean

                    real_mae_list.append(masked_mae(pred_real, true_real, m_e).item())
                    real_mse_list.append(masked_mse(pred_real, true_real, m_e).item())

            def safe_mean(lst):
                vals = [x for x in lst if x == x]
                return float(np.mean(vals)) if vals else 0.0

            final_norm_mae = safe_mean(norm_mae_list)
            final_norm_rmse = float(np.sqrt(safe_mean(norm_mse_list)))
            final_real_mae = safe_mean(real_mae_list)
            final_real_rmse = float(np.sqrt(safe_mean(real_mse_list)))

            eval_str = (
                f"Eval @ Epoch {epoch}:\n"
                f"  [Norm]  MAE={final_norm_mae:.4f}  RMSE={final_norm_rmse:.4f}\n"
                f"  [Real]  MAE={final_real_mae:.4f}  RMSE={final_real_rmse:.4f}")
            print(eval_str)
            logging.info(eval_str)

            if final_real_mae < best_real_mae:
                best_real_mae = final_real_mae
                save_path = os.path.join(base_dir, f"best_{config.dataset}_{config.task_name}.pt")
                torch.save(model, save_path)
                msg = f"*** Best model saved (Real MAE={best_real_mae:.4f}) -> {save_path}"
                print(msg)
                logging.info(msg)


if __name__ == '__main__':
    main()
