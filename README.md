# STAD-Imputer

**Spatio-Temporal Adaptive Diffusion Imputer for Coastal Remote Sensing**

> Remote-sensing variables are essential for monitoring coastal marine ecosystems,
> yet frequent cloud contamination often causes extreme data sparsity with
> missing rates exceeding 90%. **STAD-Imputer** is a unified conditional diffusion
> framework that addresses this challenge through three adaptive modules –
> **ATI**, **ANA**, and **AHM** – designed to jointly handle temporal scale
> mismatch, neighborhood heterogeneity, and spatial sparsity.

---

## 1. Method Overview

STAD-Imputer integrates three complementary modules inside a conditional DDPM
backbone:

| Module | Full name | Role |
|--------|-----------|------|
| **ATI** | Adaptive Temporal Integrator | Scale-aware integration of dilated temporal branches with gated expert routing |
| **ANA** | Adaptive Neighborhood Aggregator | Dual-path aggregation combining physical proximity and prototype-guided semantic similarity |
| **AHM** | Adaptive Heterogeneity Modulator | Multi-scale spatial aggregation with sparse expert modulation for spatial heterogeneity |

### Architecture

![STAD-Imputer Architecture](assets/frame.png)

> The full-resolution diagram is also available in PDF:
> [`assets/frame.pdf`](assets/frame.pdf)

---

## 2. Repository Structure

```
STAD-Imputer/
├── assets/
│   ├── frame.pdf                # Network architecture figure (vector)
│   └── frame.png                # Network architecture figure (raster)
├── dataset/
│   ├── dataset.py               # CoastalImputationDataset (sst4 / par / chla)
│   └── traffic_dataset.py       # AQI36 / METR-LA / PEMS-BAY loaders & adj
├── model/
│   ├── modules.py               # ATI, ANA, AHM module definitions
│   ├── stad_imputer.py          # STAD_Imputer main model + diffusion wrapper
│   ├── csdi_traffic.py          # CSDI baseline (traffic-adapted)
│   └── stimp_traffic.py         # STIMP baseline (traffic-adapted)
├── scripts/
│   ├── train_aqi36.sh           # STAD-Imputer on AQI36 / METR-LA / PEMS-BAY
│   ├── train_metrla.sh
│   ├── train_pemsbay.sh
│   ├── train_csdi_*.sh          # CSDI baseline scripts
│   └── train_stimp_*.sh         # STIMP baseline scripts
├── train.py                     # Entry point for coastal datasets
├── train_traffic.py             # Entry point for AQI36 / METR-LA / PEMS-BAY
├── train_csdi_traffic.py        # CSDI baseline trainer
├── train_stimp_traffic.py       # STIMP baseline trainer
├── utils.py                     # Metrics & helpers
└── requirements.txt
```

---

## 3. Installation

```bash
# Recommended: create a fresh conda environment
conda create -n stad python=3.9 -y
conda activate stad

# Install dependencies
pip install -r requirements.txt
```

Main dependencies: `torch`, `numpy`, `pandas`, `scikit-learn`, `tqdm`,
`linear-attention-transformer`.

---

## 4. Data Preparation

### 4.1 Coastal remote-sensing datasets

Each variable directory under `data_root/<area>/` should contain:

```
<variable>.npy                # raw data array (T, H, W)
is_sea.npy                    # binary sea mask (H, W)
adj.npy                       # adjacency matrix (N, N)
mean.npy / std.npy            # full-grid statistics
mean_init.npy / std_init.npy  # sea-point statistics
max.npy / min.npy             # per-point value bounds
```

Supported variables: `sst4`, `par`, `sst11`, `chla`.

### 4.2 Traffic / air-quality datasets

- **AQI36**: Hangzhou air-quality PM2.5, 36 stations.
- **METR-LA**: Los Angeles traffic speed, 207 sensors.
- **PEMS-BAY**: Bay-area traffic speed, 325 sensors.

Refer to [`dataset/traffic_dataset.py`](dataset/traffic_dataset.py) for the
expected file layout. Adjacency matrices are generated automatically.

---

## 5. Quick Start

### 5.1 Coastal datasets (SST4 / PAR / Chl-a)

```bash
# SST4 (Pearl River Estuary)
python train.py \
    --data_root /path/to/zone_sst4_data \
    --area PRE \
    --datasets_type sst4 \
    --epochs 500 \
    --batch_size 1 \
    --missing_ratio 0.9 \
    --test_freq 500

# PAR
python train.py --data_root /path/to/zone_par_data \
                --area PRE --datasets_type par --epochs 500

# Chlorophyll-a
python train.py --data_root /path/to/zone_chla_data \
                --area PRE --datasets_type chla --epochs 500
```

### 5.2 Traffic / air-quality datasets

```bash
# STAD-Imputer
bash scripts/train_aqi36.sh
bash scripts/train_metrla.sh
bash scripts/train_pemsbay.sh
```

### 5.3 Baseline methods (CSDI / STIMP)

```bash
# CSDI baseline
bash scripts/train_csdi_aqi36.sh
bash scripts/train_csdi_metrla.sh
bash scripts/train_csdi_pemsbay.sh

# STIMP baseline
bash scripts/train_stimp_aqi36.sh
bash scripts/train_stimp_metrla.sh
bash scripts/train_stimp_pemsbay.sh
```

All baselines share the same data split, missing pattern
(`missing_pattern=point`, `missing_ratio=0.9`) and evaluation protocol for a
fair comparison.

---

## 6. Key Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--missing_ratio` | 0.9 | Fraction of observations randomly masked during training |
| `--num_steps` | 50 | Diffusion denoising steps |
| `--num_samples` | 10 | Monte-Carlo samples at inference |
| `--ATI_tcn_layers` | 2 | Number of stacked ATI blocks |
| `--ATI_dilation_choices` | 1,2,4,8 | Dilation rates for local TCN experts |
| `--ANA_k_phys` | 8 | Physical neighbour count |
| `--ANA_k_feat` | 8 | Semantic neighbour count |
| `--ANA_num_prototypes` | 32 | Number of learnable prototype vectors |
| `--AHM_num_experts` | 8 | Number of sparse spatial experts |
| `--AHM_top_k` | 3 | Top-k expert routing |
| `--AHM_num_scales` | 3 | Graph aggregation scales |
| `--balance_weight` | 0.01 | MoE load-balance loss coefficient |

---

## 7. Evaluation Metrics

At each test epoch the following metrics are reported:

- **[Norm]** MAE, MSE (normalized space)
- **[Real]** MAE, RMSE, MAPE (physical units)
- **[Real]** R², SSIM, CRPS

---

## 8. Citation

If you find this work useful, please cite:

```bibtex
@article{stad_imputer,
  title   = {STAD-Imputer: Spatio-Temporal Adaptive Diffusion Imputer for
             Coastal Remote Sensing Data},
  year    = {2025}
}
```

---

## 9. License

This project is released for academic research purposes.
