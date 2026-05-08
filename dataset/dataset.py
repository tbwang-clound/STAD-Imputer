"""
Dataset for STAD-Imputer coastal remote sensing imputation.

Supports SST4, PAR, SST11, and Chl-a (chla) variables.
Data split follows a block-split strategy aligned with LLM4HRSI.
"""

import warnings
import numpy as np
import os
import pickle
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Scalers
# ---------------------------------------------------------------------------

class StandardScaler:
    """Z-score normalization."""

    def transform(self, x, mean, std):
        return (x - mean) / (std + 1e-12)

    def inverse_transform(self, x, mean, std):
        return x * std + mean


class LogScaler:
    """Log10 normalization (used for Chl-a)."""

    def transform(self, x):
        return np.log10(x + 1e-10)

    def inverse_transform(self, x):
        return 10 ** x


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CoastalImputationDataset(Dataset):
    """
    Sliding-window dataset for coastal remote sensing imputation.

    Directory layout expected under ``data_root / area /``::

        <variable>.npy     - raw data array, shape (T, H, W)
        is_sea.npy         - binary sea mask,  shape (H, W)
        adj.npy            - adjacency matrix,  shape (N, N)
        mean.npy / std.npy - full-grid statistics for visualisation
        mean_init.npy / std_init.npy - sea-point statistics for normalisation
        max.npy / min.npy  - per-point bounds used to clamp model output

    Args:
        config:  Namespace with at least:
                 data_root, area, datasets_type, datasets_standard_type,
                 in_len, out_len, missing_ratio
        mode (str): 'train' or 'test'
    """

    def __init__(self, config, mode="train"):
        super().__init__()
        self.config = config
        self.mode = mode

        # ---- Paths ----
        self.data_dir = os.path.join(config.data_root, config.area)
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        data_file = os.path.join(self.data_dir, f"{config.datasets_type}.npy")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        self.data_file_path = data_file

        self.cache_path = os.path.join(
            self.data_dir,
            f"{config.datasets_type}_{config.datasets_standard_type}"
            f"_missing_{config.missing_ratio}"
            f"_in_{config.in_len}_out_{config.out_len}.pk"
        )

        print(f"[Dataset] variable={config.datasets_type}  mode={mode}  "
              f"dir={self.data_dir}")

        self.in_len = config.in_len
        self.out_len = config.out_len

        # ---- Auxiliary statistics ----
        self.adj = np.load(os.path.join(self.data_dir, "adj.npy"))
        self.area_mask = np.load(os.path.join(self.data_dir, "is_sea.npy"))

        if config.datasets_type != 'chla':
            mean_raw = np.load(os.path.join(self.data_dir, "mean_init.npy"))
            std_raw = np.load(os.path.join(self.data_dir, "std_init.npy"))
            self.mean = mean_raw[self.area_mask.astype(bool)]
            self.std = std_raw[self.area_mask.astype(bool)]
        else:
            self.mean = None
            self.std = None

        max_raw = np.load(os.path.join(self.data_dir, "max.npy"))
        min_raw = np.load(os.path.join(self.data_dir, "min.npy"))
        self.max = max_raw[self.area_mask.astype(bool)]
        self.min = min_raw[self.area_mask.astype(bool)]

        # ---- Scaler ----
        if config.datasets_type == 'chla':
            self.scaler = LogScaler()
            self.scaler_mean = None
            self.scaler_std = None
        elif config.datasets_type in ['par', 'sst4', 'sst', 'sst11']:
            self.scaler = StandardScaler()
            self.scaler_mean = mean_raw
            self.scaler_std = std_raw
        else:
            raise ValueError(f"Unsupported datasets_type: {config.datasets_type}")

        # ---- Load / generate sliding-window samples ----
        raw_data, raw_mask = self._load_raw()
        self._build_or_load_cache(raw_data, raw_mask)

        # ---- Spatial filter: keep sea points only ----
        sea = self.area_mask.astype(bool)
        self.datas = self.datas[:, :, :, sea]
        self.data_ob_masks = self.data_ob_masks[:, :, :, sea]
        self.data_gt_masks = self.data_gt_masks[:, :, :, sea]
        self.labels = self.labels[:, :, :, sea]
        self.label_ob_masks = self.label_ob_masks[:, :, :, sea]

        # ---- Block-split train / test ----
        self._split(config)

    # ------------------------------------------------------------------

    def _load_raw(self):
        print(f"[Dataset] Loading {self.data_file_path}")
        data_raw = np.load(self.data_file_path)
        data_raw = data_raw[:, np.newaxis]          # (T, 1, H, W)
        mask_raw = np.isfinite(data_raw).astype(np.float32)

        if self.config.datasets_standard_type == 'repete':
            if isinstance(self.scaler, StandardScaler):
                data_t = self.scaler.transform(data_raw,
                                               self.scaler_mean, self.scaler_std)
            else:
                data_t = self.scaler.transform(data_raw)
        else:
            raise ValueError("Only 'repete' normalization is supported.")

        data_filled = np.nan_to_num(data_t, nan=0.)
        return data_filled, mask_raw

    def _build_or_load_cache(self, oral_data, oral_mask):
        if os.path.isfile(self.cache_path):
            print(f"[Dataset] Loading cache: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                (self.datas, self.data_ob_masks, self.data_gt_masks,
                 self.labels, self.label_ob_masks) = pickle.load(f)
            return

        print(f"[Dataset] Building cache: {self.cache_path}")
        total_samples = len(oral_data) - self.in_len - self.out_len + 1
        datas, ob_masks, gt_masks, labels, lb_masks = [], [], [], [], []

        for idx in range(total_samples):
            data = oral_data[idx: idx + self.in_len]
            ob_mask = oral_mask[idx: idx + self.in_len]
            label = oral_data[idx + self.in_len: idx + self.in_len + self.out_len]
            lb_mask = oral_mask[idx + self.in_len: idx + self.in_len + self.out_len]

            # Generate ground-truth mask by randomly hiding some observations
            masks = ob_mask.reshape(-1).copy()
            obs_idx = np.where(masks)[0].tolist()
            miss_idx = np.random.choice(
                obs_idx, int(len(obs_idx) * self.config.missing_ratio), replace=False
            )
            masks[miss_idx] = False
            gt_mask = masks.reshape(ob_mask.shape)

            datas.append(data)
            ob_masks.append(ob_mask)
            gt_masks.append(gt_mask)
            labels.append(label)
            lb_masks.append(lb_mask)

        self.datas = np.array(datas).astype("float32")
        self.data_ob_masks = np.array(ob_masks).astype("float32")
        self.data_gt_masks = np.array(gt_masks).astype("float32")
        self.labels = np.array(labels).astype("float32")
        self.label_ob_masks = np.array(lb_masks).astype("float32")

        with open(self.cache_path, "wb") as f:
            pickle.dump([self.datas, self.data_ob_masks, self.data_gt_masks,
                         self.labels, self.label_ob_masks], f)
        print(f"[Dataset] Cache saved ({total_samples} samples).")

    def _split(self, config):
        """Block-split aligned with LLM4HRSI."""
        boundaries = {
            'chla': 648, 'par': 556, 'sst4': 535,
            'sst11': 540, 'sst': 540,
        }
        raw_bound = boundaries[config.datasets_type]
        train_end = raw_bound - self.in_len - self.out_len + 1
        test_start = raw_bound

        print(f"[Dataset] Block-split | raw_bound={raw_bound} "
              f"train=[0,{train_end}) test=[{test_start},end) "
              f"gap={test_start - train_end}")

        if self.mode == "train":
            train_end = min(train_end, len(self.datas))
            self.datas = self.datas[:train_end]
            self.data_ob_masks = self.data_ob_masks[:train_end]
            self.data_gt_masks = self.data_gt_masks[:train_end]
            self.labels = self.labels[:train_end]
            self.label_ob_masks = self.label_ob_masks[:train_end]
            print(f"[Dataset] Train samples: {len(self.datas)}")
        else:
            if test_start < len(self.datas):
                self.datas = self.datas[test_start:]
                self.data_ob_masks = self.data_ob_masks[test_start:]
                self.data_gt_masks = self.data_gt_masks[test_start:]
                self.labels = self.labels[test_start:]
                self.label_ob_masks = self.label_ob_masks[test_start:]
            else:
                self.datas = self.datas[0:0]
            print(f"[Dataset] Test samples: {len(self.datas)}")

    # ------------------------------------------------------------------

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, index):
        return (self.datas[index], self.data_ob_masks[index],
                self.data_gt_masks[index], self.labels[index],
                self.label_ob_masks[index])
