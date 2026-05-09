#!/usr/bin/env python3
"""
DINEOF Baseline for STAD-Imputer (Coastal Datasets)

Migrated from STIMP-release/imputation/train_dineof_per_timestep.py
Adapted to use STAD-Imputer's dataset interface.
"""
import argparse
import torch
import os
import sys
import logging
import time
from tqdm import tqdm
import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import svd as dense_svd
from sklearn.base import BaseEstimator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataset import CoastalImputationDataset
from utils import check_dir, masked_mae, masked_mse, seed_everything


# ==================== DINEOF Model ====================
def rectify_tensor(tensor):
    """Reshape (H, W, T) -> (H*W, T)"""
    H, W, T = tensor.shape
    return tensor.reshape(H * W, T)


def center_mat(mat):
    nan_mask = np.isnan(mat)
    temp_mat = mat.copy()
    temp_mat[nan_mask] = 0
    m0 = temp_mat.mean(axis=0)
    for i in range(temp_mat.shape[0]):
        temp_mat[i, :] -= m0
    m1 = temp_mat.mean(axis=1)
    for i in range(temp_mat.shape[1]):
        temp_mat[:, i] -= m1
    temp_mat[nan_mask] = np.nan
    return temp_mat, m0, m1


def decenter_mat(mat, m0, m1):
    temp_mat = mat.copy()
    for i in range(temp_mat.shape[0]):
        temp_mat[i, :] += m0
    for i in range(temp_mat.shape[1]):
        temp_mat[:, i] += m1
    return temp_mat


class DINEOF(BaseEstimator):
    def __init__(self, K=10, nitemax=300, toliter=1e-5, tol=1e-8):
        self.K = K
        self.nitemax = nitemax
        self.toliter = toliter
        self.tol = tol

    def fit(self, mat):
        if mat.ndim > 2:
            mat = rectify_tensor(mat)
        mat, *means = center_mat(mat)
        nan_mask = np.isnan(mat)
        non_nan_mask = ~nan_mask
        mat[nan_mask] = 0
        conv_error = 0
        for i in range(self.nitemax):
            try:
                u, s, vt = svds(mat, k=self.K, tol=self.tol)
            except Exception:
                u, s, vt = dense_svd(mat, full_matrices=False)
                u, s, vt = u[:, :self.K], s[:self.K], vt[:self.K, :]
            mat_hat = u @ np.diag(s) @ vt
            mat_hat[non_nan_mask] = mat[non_nan_mask]
            std_val = mat[non_nan_mask].std()
            if std_val == 0:
                std_val = 1e-6
            new_conv_error = np.sqrt(np.mean((mat_hat[nan_mask] - mat[nan_mask])**2)) / std_val
            if abs(new_conv_error - conv_error) < self.toliter or new_conv_error <= self.toliter:
                break
            conv_error = new_conv_error
            mat = mat_hat
        mat = decenter_mat(mat, *means)
        self.reconstructed_tensor = mat


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--area', type=str, default='PRE')
    parser.add_argument('--datasets_type', type=str, default='chla')
    parser.add_argument('--missing_ratio', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--model_K', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    base_dir = f"./checkpoints/baselines/dineof_{args.area}_{args.datasets_type}/"
    check_dir(base_dir)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logging.basicConfig(level=logging.INFO,
                        filename=os.path.join(base_dir, f'{timestamp}_mr{args.missing_ratio}.log'),
                        filemode='a', format='%(asctime)s - %(message)s')

    # Dataset
    test_dataset = CoastalImputationDataset(
        data_root=args.data_root, area=args.area,
        datasets_type=args.datasets_type, mode='test',
        missing_ratio=args.missing_ratio, missing_pattern='random'
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    scaler_mean = torch.from_numpy(test_dataset.mean).float() if args.datasets_type != 'chla' else None
    scaler_std = torch.from_numpy(test_dataset.std).float() if args.datasets_type != 'chla' else None

    model = DINEOF(K=args.model_K)
    norm_mae_list, norm_mse_list, real_mae_list, real_mse_list = [], [], [], []

    for datas, data_ob_masks, data_gt_masks in tqdm(test_loader):
        B, T, C, H, W = datas.shape
        impute_list = []
        for t in range(T):
            data = datas[:, t, 0].squeeze().cpu().numpy()
            tmp = np.where(data_gt_masks[:, t, 0].cpu().squeeze().numpy() == 0, np.nan, data)
            tmp = np.where(data_ob_masks[:, t, 0].cpu().squeeze().numpy() == 0, np.nan, tmp)
            model.fit(tmp)
            impute_list.append(torch.from_numpy(model.reconstructed_tensor).view(1, 1, H, W))
        imputed = torch.cat(impute_list, dim=1)  # [1, T, H, W]
        mask = (data_ob_masks - data_gt_masks)[:, :, 0]

        pred_norm = imputed.detach()
        true_norm = datas[:, :, 0]
        norm_mae_list.append(masked_mae(pred_norm, true_norm, mask).item())
        norm_mse_list.append(masked_mse(pred_norm, true_norm, mask).item())

        if args.datasets_type == 'chla':
            pred_real = torch.pow(10, pred_norm)
            true_real = torch.pow(10, true_norm)
        else:
            pred_real = pred_norm * scaler_std + scaler_mean
            true_real = true_norm * scaler_std + scaler_mean
        real_mae_list.append(masked_mae(pred_real, true_real, mask).item())
        real_mse_list.append(masked_mse(pred_real, true_real, mask).item())

    final_norm_mae = np.mean([x for x in norm_mae_list if x != 0])
    final_norm_mse = np.mean([x for x in norm_mse_list if x != 0])
    final_real_mae = np.mean([x for x in real_mae_list if x != 0])
    final_real_rmse = np.sqrt(np.mean([x for x in real_mse_list if x != 0]))

    log = (f"\n=== DINEOF Results ({args.datasets_type}) ===\n"
           f" [Norm] MAE: {final_norm_mae:.4f}, MSE: {final_norm_mse:.4f}\n"
           f" [Real] MAE: {final_real_mae:.4f}, RMSE: {final_real_rmse:.4f}")
    print(log)
    logging.info(log)


if __name__ == '__main__':
    main()
