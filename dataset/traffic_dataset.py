"""
Traffic dataset loaders for STAD-Imputer.

Supports three public traffic / air-quality benchmarks:
  - METR-LA  (207 sensors, speed)
  - PEMS-BAY (325 sensors, speed)
  - AQI-36   (36 stations, PM2.5)

Each __getitem__ returns a tuple
    (observed_data, observed_mask, gt_mask, cond_mask)
of shape (T, 1, N) so they are compatible with the coastal dataset loader
and the STAD-Imputer interface which expects (B, T, K, N) after collation.

Data files expected:
  METR-LA:
      data/metr_la/metr_la.h5
      data/metr_la/metr_la_dist.npy
      data/metr_la/metr_meanstd.pk          (auto-generated if missing)
  PEMS-BAY:
      data/pems_bay/pems_bay.h5
      data/pems_bay/pems_bay_dist.npy
      data/pems_bay/pems_meanstd.pk          (auto-generated if missing)
  AQI-36:
      data/pm25/SampleData/pm25_ground.txt
      data/pm25/SampleData/pm25_missing.txt
      data/pm25/SampleData/pm25_latlng.txt
      data/pm25/pm25_meanstd.pk              (auto-generated if missing)
"""

import os
import pickle
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Shared mask utilities (aligned with PriSTI)
# ---------------------------------------------------------------------------

def sample_block_mask(shape, p=0.0015, p_noise=0.05,
                      min_seq=12, max_seq=48, seed=9101112):
    """Generate a reproducible block-missing mask (uint8)."""
    rng = np.random.default_rng(seed)
    mask = rng.random(shape) < p
    for col in range(mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq + int(rng.integers(max_seq - min_seq)) \
            if max_seq > min_seq else min_seq
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(np.clip(idxs_ext, 0, shape[0] - 1))
        mask[idxs, col] = True
    mask = mask | (rng.random(mask.shape) < p_noise)
    return mask.astype('uint8')


def get_randmask(observed_mask, missing_ratio=0.1):
    """Randomly mask a fraction of observed positions (returns float tensor)."""
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    flat = rand_for_mask.reshape(-1)
    num_observed = observed_mask.sum().item()
    num_masked = round(num_observed * missing_ratio)
    if num_masked > 0:
        flat[flat.topk(num_masked).indices] = -1
    cond_mask = (flat.reshape(observed_mask.shape) > 0).float()
    return cond_mask


def get_block_mask(observed_mask, block_len=12):
    """Randomly blank one contiguous time block per node."""
    T, N = observed_mask.shape
    cond_mask = observed_mask.clone()
    for n in range(N):
        obs_t = (observed_mask[:, n] > 0).nonzero(as_tuple=True)[0]
        if len(obs_t) < block_len:
            continue
        start = obs_t[torch.randint(len(obs_t) - block_len, (1,)).item()].item()
        cond_mask[start: start + block_len, n] = 0.
    return cond_mask.float()


# ---------------------------------------------------------------------------
# Adjacency matrix helpers
# ---------------------------------------------------------------------------

def _build_adj_from_dist(dist_npy, thr=0.1):
    """Gaussian-kernel adjacency (no self-loop) from a distance matrix."""
    dist = np.load(dist_npy)
    finite = dist.reshape(-1)
    finite = finite[~np.isinf(finite)]
    sigma = finite.std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < thr] = 0.
    np.fill_diagonal(adj, 0.)
    return adj.astype(np.float32)


def _build_adj_aqi36(latlng_csv, thr=0.1):
    """Build AQI-36 adjacency from lat/lon CSV."""
    from sklearn.metrics.pairwise import haversine_distances
    df = pd.read_csv(latlng_csv)[['latitude', 'longitude']]
    latlon = np.radians(df.values)
    dist = haversine_distances(latlon) * 6371.0088  # km
    theta = np.std(dist[:36, :36])
    adj = np.exp(-np.square(dist / theta))
    adj[adj < thr] = 0.
    np.fill_diagonal(adj, 0.)
    return adj[:36, :36].astype(np.float32)


# ---------------------------------------------------------------------------
# METR-LA
# ---------------------------------------------------------------------------

class MetrLADataset(Dataset):
    """
    METR-LA traffic speed imputation dataset.

    Returns (observed_data, observed_mask, gt_mask, cond_mask)
    each of shape (T, 1, N)  where T = eval_length, N = 207.
    """

    NUM_NODES = 207
    TOTAL_STEPS = 34272

    def __init__(self, data_root, mode="train", eval_length=24,
                 val_len=0.1, test_len=0.2,
                 missing_pattern='block', missing_ratio=0.1):
        super().__init__()
        self.eval_length = eval_length
        self.mode = mode
        self.missing_ratio = missing_ratio
        self.missing_pattern = missing_pattern
        self.data_root = data_root

        hdf_path = os.path.join(data_root, "metr_la", "metr_la.h5")
        meanstd_path = os.path.join(data_root, "metr_la", "metr_meanstd.pk")

        # ---- Load / compute statistics ----
        df = pd.read_hdf(hdf_path)
        if not os.path.exists(meanstd_path):
            n = int(len(df) * 0.7)
            mean = np.mean(df.values[:n], axis=0)
            std = np.std(df.values[:n], axis=0)
            with open(meanstd_path, 'wb') as f:
                pickle.dump((mean, std), f)
        with open(meanstd_path, 'rb') as f:
            self.train_mean, self.train_std = pickle.load(f)

        # ---- Masks ----
        ob_mask = (df.values != 0.).astype('uint8')        # (T_total, N)
        eval_mask = sample_block_mask(
            shape=(self.TOTAL_STEPS, self.NUM_NODES),
            p=0.0015, p_noise=0.05, min_seq=12, max_seq=48)
        gt_mask = (1 - (eval_mask | (1 - ob_mask))).astype('uint8')

        # ---- Normalise ----
        c_data = ((df.fillna(0.).values - self.train_mean)
                  / (self.train_std + 1e-8)) * ob_mask

        # ---- Split ----
        val_start = int((1 - val_len - test_len) * len(df))
        test_start = int((1 - test_len) * len(df))

        if mode == 'train':
            self.observed_data = c_data[:val_start]
            self.observed_mask = ob_mask[:val_start]
            self.gt_mask = gt_mask[:val_start]
        elif mode == 'valid':
            self.observed_data = c_data[val_start:test_start]
            self.observed_mask = ob_mask[val_start:test_start]
            self.gt_mask = gt_mask[val_start:test_start]
        else:  # test
            self.observed_data = c_data[test_start:]
            self.observed_mask = ob_mask[test_start:]
            self.gt_mask = gt_mask[test_start:]

        self._build_index()

    def _build_index(self):
        L = len(self.observed_data)
        T = self.eval_length
        if self.mode == 'test':
            n = L // T
            self.use_index = list(np.arange(0, n * T, T))
            self.cut_length = [0] * n
            if L % T != 0:
                self.use_index.append(L - T)
                self.cut_length.append(T - L % T)
        else:
            self.use_index = list(np.arange(L - T + 1))
            self.cut_length = [0] * len(self.use_index)

    def __len__(self):
        return len(self.use_index)

    def __getitem__(self, org_index):
        idx = self.use_index[org_index]
        T = self.eval_length
        ob_data = self.observed_data[idx: idx + T]   # (T, N)
        ob_mask = self.observed_mask[idx: idx + T]
        gt_mask = self.gt_mask[idx: idx + T]

        ob_mask_t = torch.tensor(ob_mask, dtype=torch.float32)

        if self.mode != 'train':
            cond_mask = torch.tensor(gt_mask, dtype=torch.float32)
        else:
            if self.missing_pattern == 'block':
                cond_mask = get_block_mask(ob_mask_t)
            else:
                cond_mask = get_randmask(ob_mask_t, self.missing_ratio)

        # Expand channel dim: (T, N) -> (T, 1, N)
        return (
            torch.tensor(ob_data, dtype=torch.float32).unsqueeze(1),
            ob_mask_t.unsqueeze(1),
            torch.tensor(gt_mask, dtype=torch.float32).unsqueeze(1),
            cond_mask.unsqueeze(1),
        )


# ---------------------------------------------------------------------------
# PEMS-BAY
# ---------------------------------------------------------------------------

class PemsBayDataset(Dataset):
    """
    PEMS-BAY traffic speed imputation dataset.

    Returns (observed_data, observed_mask, gt_mask, cond_mask)
    each of shape (T, 1, N)  where N = 325.
    """

    NUM_NODES = 325
    TOTAL_STEPS = 52116

    def __init__(self, data_root, mode="train", eval_length=24,
                 val_len=0.1, test_len=0.2,
                 missing_pattern='block', missing_ratio=0.1):
        super().__init__()
        self.eval_length = eval_length
        self.mode = mode
        self.missing_ratio = missing_ratio
        self.missing_pattern = missing_pattern
        self.data_root = data_root

        hdf_path = os.path.join(data_root, "pems_bay", "pems_bay.h5")
        meanstd_path = os.path.join(data_root, "pems_bay", "pems_meanstd.pk")

        df = pd.read_hdf(hdf_path)
        if not os.path.exists(meanstd_path):
            n = int(len(df) * 0.7)
            mean = np.mean(df.values[:n], axis=0)
            std = np.std(df.values[:n], axis=0)
            with open(meanstd_path, 'wb') as f:
                pickle.dump((mean, std), f)
        with open(meanstd_path, 'rb') as f:
            self.train_mean, self.train_std = pickle.load(f)

        ob_mask = (df.values != 0.).astype('uint8')
        eval_mask = sample_block_mask(
            shape=(self.TOTAL_STEPS, self.NUM_NODES),
            p=0.0015, p_noise=0.05, min_seq=12, max_seq=48)
        gt_mask = (1 - (eval_mask | (1 - ob_mask))).astype('uint8')

        c_data = ((df.fillna(0.).values - self.train_mean)
                  / (self.train_std + 1e-8)) * ob_mask

        val_start = int((1 - val_len - test_len) * len(df))
        test_start = int((1 - test_len) * len(df))

        if mode == 'train':
            self.observed_data = c_data[:val_start]
            self.observed_mask = ob_mask[:val_start]
            self.gt_mask = gt_mask[:val_start]
        elif mode == 'valid':
            self.observed_data = c_data[val_start:test_start]
            self.observed_mask = ob_mask[val_start:test_start]
            self.gt_mask = gt_mask[val_start:test_start]
        else:
            self.observed_data = c_data[test_start:]
            self.observed_mask = ob_mask[test_start:]
            self.gt_mask = gt_mask[test_start:]

        self._build_index()

    def _build_index(self):
        L = len(self.observed_data)
        T = self.eval_length
        if self.mode == 'test':
            n = L // T
            self.use_index = list(np.arange(0, n * T, T))
            self.cut_length = [0] * n
            if L % T != 0:
                self.use_index.append(L - T)
                self.cut_length.append(T - L % T)
        else:
            self.use_index = list(np.arange(L - T + 1))
            self.cut_length = [0] * len(self.use_index)

    def __len__(self):
        return len(self.use_index)

    def __getitem__(self, org_index):
        idx = self.use_index[org_index]
        T = self.eval_length
        ob_data = self.observed_data[idx: idx + T]
        ob_mask = self.observed_mask[idx: idx + T]
        gt_mask = self.gt_mask[idx: idx + T]

        ob_mask_t = torch.tensor(ob_mask, dtype=torch.float32)

        if self.mode != 'train':
            cond_mask = torch.tensor(gt_mask, dtype=torch.float32)
        else:
            if self.missing_pattern == 'block':
                cond_mask = get_block_mask(ob_mask_t)
            else:
                cond_mask = get_randmask(ob_mask_t, self.missing_ratio)

        return (
            torch.tensor(ob_data, dtype=torch.float32).unsqueeze(1),
            ob_mask_t.unsqueeze(1),
            torch.tensor(gt_mask, dtype=torch.float32).unsqueeze(1),
            cond_mask.unsqueeze(1),
        )


# ---------------------------------------------------------------------------
# AQI-36 (PM2.5)
# ---------------------------------------------------------------------------

class AQI36Dataset(Dataset):
    """
    AQI-36 air quality (PM2.5) imputation dataset.

    Month split identical to PriSTI:
      train = months [1,2,4,5,7,8,10,11]
      valid = months [2,5,8,11]  (last val_len fraction)
      test  = months [3,6,9,12]

    Returns (observed_data, observed_mask, gt_mask, cond_mask)
    each of shape (T, 1, N)  where T = eval_length, N = 36.
    """

    NUM_NODES = 36

    def __init__(self, data_root, mode="train", eval_length=36,
                 val_len=0.1, missing_ratio=0.1, target_strategy='random'):
        super().__init__()
        self.eval_length = eval_length
        self.mode = mode
        self.missing_ratio = missing_ratio
        self.target_strategy = target_strategy
        self.data_root = data_root

        ground_path = os.path.join(
            data_root, "pm25", "SampleData", "pm25_ground.txt")
        missing_path = os.path.join(
            data_root, "pm25", "SampleData", "pm25_missing.txt")
        meanstd_path = os.path.join(data_root, "pm25", "pm25_meanstd.pk")

        df = pd.read_csv(ground_path, index_col="datetime", parse_dates=True)
        df_gt = pd.read_csv(missing_path, index_col="datetime", parse_dates=True)

        # ---- Statistics (computed on train months) ----
        if not os.path.exists(meanstd_path):
            train_months = [1, 2, 4, 5, 7, 8, 10, 11]
            train_df = df[df.index.month.isin(train_months)]
            mean = train_df.fillna(0.).mean(axis=0).values
            std = train_df.fillna(0.).std(axis=0).values
            std[std < 1e-8] = 1.0
            with open(meanstd_path, 'wb') as f:
                pickle.dump((mean, std), f)
        with open(meanstd_path, 'rb') as f:
            self.train_mean, self.train_std = pickle.load(f)

        # ---- Month lists ----
        if mode == 'train':
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]
        elif mode == 'valid':
            month_list = [2, 5, 8, 11]
        else:
            month_list = [3, 6, 9, 12]

        self.observed_data = []
        self.observed_mask = []
        self.gt_mask = []
        self.index_month = []
        self.position_in_month = []
        self.use_index = []
        self.cut_length = []

        for i, m in enumerate(month_list):
            cur_df = df[df.index.month == m]
            cur_df_gt = df_gt[df_gt.index.month == m]

            if mode == 'train' and m in [2, 5, 8, 11]:
                cut = int(val_len * len(cur_df))
                cur_df = cur_df.iloc[:-cut]
                cur_df_gt = cur_df_gt.iloc[:-cut]
            elif mode == 'valid':
                cut = int(val_len * len(cur_df))
                cur_df = cur_df.iloc[-cut:]
                cur_df_gt = cur_df_gt.iloc[-cut:]

            c_mask = (1 - cur_df.isnull().values).astype('uint8')
            c_gt_mask = (1 - cur_df_gt.isnull().values).astype('uint8')
            c_data = ((cur_df.fillna(0.).values - self.train_mean)
                      / (self.train_std + 1e-8)) * c_mask

            self.observed_data.append(c_data)
            self.observed_mask.append(c_mask)
            self.gt_mask.append(c_gt_mask)

            cur_len = len(cur_df) - eval_length + 1
            last = len(self.index_month)
            self.index_month += [i] * cur_len
            self.position_in_month += list(range(cur_len))

            if mode == 'test':
                n_s = len(cur_df) // eval_length
                c_idx = np.arange(last, last + eval_length * n_s, eval_length)
                self.use_index += c_idx.tolist()
                self.cut_length += [0] * len(c_idx)
                if len(cur_df) % eval_length != 0:
                    self.use_index.append(len(self.index_month) - 1)
                    self.cut_length.append(
                        eval_length - len(cur_df) % eval_length)

        if mode != 'test':
            self.use_index = list(range(len(self.index_month)))
            self.cut_length = [0] * len(self.use_index)

    def __len__(self):
        return len(self.use_index)

    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        c_month = self.index_month[index]
        c_pos = self.position_in_month[index]
        T = self.eval_length

        ob_data = self.observed_data[c_month][c_pos: c_pos + T]  # (T, N)
        ob_mask = self.observed_mask[c_month][c_pos: c_pos + T]
        gt_mask = self.gt_mask[c_month][c_pos: c_pos + T]

        ob_mask_t = torch.tensor(ob_mask, dtype=torch.float32)

        if self.mode != 'train':
            cond_mask = torch.tensor(gt_mask, dtype=torch.float32)
        else:
            cond_mask = get_randmask(ob_mask_t, self.missing_ratio)

        return (
            torch.tensor(ob_data, dtype=torch.float32).unsqueeze(1),
            ob_mask_t.unsqueeze(1),
            torch.tensor(gt_mask, dtype=torch.float32).unsqueeze(1),
            cond_mask.unsqueeze(1),
        )


# ---------------------------------------------------------------------------
# Adjacency matrix factory
# ---------------------------------------------------------------------------

def get_adj(dataset_name, data_root):
    """
    Build / load the adjacency matrix for the given traffic dataset.

    Returns:
        adj: np.ndarray float32, shape (N, N), row-normalised in [0,1]
    """
    if dataset_name == 'metrla':
        dist_npy = os.path.join(data_root, "metr_la", "metr_la_dist.npy")
        return _build_adj_from_dist(dist_npy)
    elif dataset_name == 'pemsbay':
        dist_npy = os.path.join(data_root, "pems_bay", "pems_bay_dist.npy")
        return _build_adj_from_dist(dist_npy)
    elif dataset_name == 'aqi36':
        latlng_csv = os.path.join(
            data_root, "pm25", "SampleData", "pm25_latlng.txt")
        return _build_adj_aqi36(latlng_csv)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         "Choose from metrla / pemsbay / aqi36")


# ---------------------------------------------------------------------------
# Unified dataloader factory
# ---------------------------------------------------------------------------

DATASET_CLS = {
    'metrla': MetrLADataset,
    'pemsbay': PemsBayDataset,
    'aqi36': AQI36Dataset,
}

DATASET_NODES = {
    'metrla': 207,
    'pemsbay': 325,
    'aqi36': 36,
}

DATASET_MEAN_STD = {
    'metrla': 'metr_la/metr_meanstd.pk',
    'pemsbay': 'pems_bay/pems_meanstd.pk',
    'aqi36': 'pm25/pm25_meanstd.pk',
}


def get_dataloader(dataset_name, data_root, batch_size,
                   eval_length=24, val_len=0.1, test_len=0.2,
                   missing_pattern='block', missing_ratio=0.1,
                   num_workers=4):
    """
    Build train / valid / test DataLoaders for a traffic dataset.

    Returns:
        train_loader, valid_loader, test_loader, train_mean, train_std
    """
    cls = DATASET_CLS[dataset_name]

    def _make(mode):
        if dataset_name == 'aqi36':
            kwargs = dict(data_root=data_root, mode=mode,
                          eval_length=eval_length, val_len=val_len,
                          missing_ratio=missing_ratio)
        else:
            kwargs = dict(data_root=data_root, mode=mode,
                          eval_length=eval_length, val_len=val_len,
                          test_len=test_len,
                          missing_pattern=missing_pattern,
                          missing_ratio=missing_ratio)
        return cls(**kwargs)

    train_ds = _make('train')
    valid_ds = _make('valid')
    test_ds = _make('test')

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    meanstd_path = os.path.join(data_root, DATASET_MEAN_STD[dataset_name])
    with open(meanstd_path, 'rb') as f:
        train_mean, train_std = pickle.load(f)

    return train_loader, valid_loader, test_loader, train_mean, train_std
