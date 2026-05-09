#!/usr/bin/env python3
"""Lin-ITP (Linear Interpolation) Baseline"""
import argparse, torch, os, sys, logging, time
from tqdm import tqdm
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataset import CoastalImputationDataset
from utils import check_dir, masked_mae, masked_mse, seed_everything

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--area', type=str, default='PRE')
    parser.add_argument('--datasets_type', type=str, default='chla')
    parser.add_argument('--missing_ratio', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    seed_everything(args.seed)
    base_dir = f"./checkpoints/baselines/lin_itp_{args.area}_{args.datasets_type}/"
    check_dir(base_dir)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logging.basicConfig(level=logging.INFO, filename=os.path.join(base_dir, f'{timestamp}_mr{args.missing_ratio}.log'), filemode='a', format='%(asctime)s - %(message)s')
    test_dataset = CoastalImputationDataset(data_root=args.data_root, area=args.area, datasets_type=args.datasets_type, mode='test', missing_ratio=args.missing_ratio, missing_pattern='random')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    scaler_mean = torch.from_numpy(test_dataset.mean).float() if args.datasets_type != 'chla' else None
    scaler_std = torch.from_numpy(test_dataset.std).float() if args.datasets_type != 'chla' else None
    norm_mae_list, norm_mse_list, real_mae_list, real_mse_list = [], [], [], []
    try:
        import torchcde
    except ImportError:
        print("Error: torchcde not installed. Run: pip install torchcde")
        sys.exit(1)
    for datas, data_ob_masks, data_gt_masks in tqdm(test_loader):
        B, T, C, H, W = datas.shape
        tmp = torch.where(data_gt_masks[:, :, 0] == 0, float('nan'), datas[:, :, 0])
        tmp = tmp.permute(0, 2, 3, 1).reshape(-1, T)
        itp = torchcde.linear_interpolation_coeffs(tmp)
        imputed = itp.reshape(B, H, W, T).permute(0, 3, 1, 2).unsqueeze(2)
        mask = (data_ob_masks - data_gt_masks)[:, :, 0]
        pred_norm = imputed[:, :, 0].detach()
        true_norm = datas[:, :, 0]
        norm_mae_list.append(masked_mae(pred_norm, true_norm, mask).item())
        norm_mse_list.append(masked_mse(pred_norm, true_norm, mask).item())
        if args.datasets_type == 'chla':
            pred_real = torch.pow(10, pred_norm); true_real = torch.pow(10, true_norm)
        else:
            pred_real = pred_norm * scaler_std + scaler_mean; true_real = true_norm * scaler_std + scaler_mean
        real_mae_list.append(masked_mae(pred_real, true_real, mask).item())
        real_mse_list.append(masked_mse(pred_real, true_real, mask).item())
    final_norm_mae = np.mean([x for x in norm_mae_list if x != 0])
    final_norm_mse = np.mean([x for x in norm_mse_list if x != 0])
    final_real_mae = np.mean([x for x in real_mae_list if x != 0])
    final_real_rmse = np.sqrt(np.mean([x for x in real_mse_list if x != 0]))
    log = f"\n=== Lin-ITP Results ({args.datasets_type}) ===\n [Norm] MAE: {final_norm_mae:.4f}, MSE: {final_norm_mse:.4f}\n [Real] MAE: {final_real_mae:.4f}, RMSE: {final_real_rmse:.4f}"
    print(log); logging.info(log)

if __name__ == '__main__':
    main()
