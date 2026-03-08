# Transformer-based Reconstruction for Electromagnetic Calorimeters
### GSoC 2026 – Technical Evaluation Task

This repository implements a regression pipeline to predict the true photon energy from calorimeter cell-level energy deposits. The dataset consists of simplified electromagnetic shower responses on a 15×15 grid (225 cells), representing isolated single-photon events with stochastic shower fluctuations and per-cell electronic noise.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Setup](#setup)
3. [Dataset](#dataset)
4. [How to Run](#how-to-run)
5. [Models](#models)
6. [Results](#results)
7. [Uncertainty Quantification](#uncertainty-quantification)
8. [Inference Performance](#inference-performance)
9. [Discussion](#discussion)
10. [Conceptual Questions](#conceptual-questions)

---

## Repository Structure

```
calorimeter-regression/
│
├── configs/
│   └── config.yaml              # all hyperparameters, seeds, paths
│
├── data/
│   └── dataset.py               # loading, splitting, normalization, DataLoader
│
├── models/
│   ├── mlp.py                   # fully-connected network
│   ├── cnn.py                   # convolutional network (treats grid as 2D image)
│   └── deepsets.py              # DeepSets (permutation-invariant, per-cell)
│
├── utils/
│   └── metrics.py               # MAE, RMSE, bias, RMS(dE/E), std(dE/E), p68
│
├── uncertainty/
│   ├── __init__.py
│   ├── bootstrap.py             # bootstrap confidence intervals
│   ├── binned_resolution.py     # resolution vs energy with error bars
│   ├── mc_dropout.py            # Monte Carlo Dropout uncertainty
│   ├── gaussian_fit.py          # Gaussian fit to dE/E distribution
│   ├── ensemble.py              # Deep Ensemble training and inference
│   └── ks_test.py               # KS test and Shapiro-Wilk normality tests
│
├── train.py                     # training loop, checkpoint saving, loss plots
├── evaluate.py                  # loads checkpoint, computes metrics, saves plots
├── infer.py                     # inference timing benchmark
├── uncertainty_runner.py        # runs all uncertainty analyses in sequence
├── generate_dataset.py          # synthetic dataset generator (see Dataset section)
├── run_all.sh                   # reproduces all results for all three models
└── requirements.txt
```

---

## Setup

```bash
git clone <repo-url>
cd calorimeter-regression
pip install -r requirements.txt
```

Requirements: `numpy`, `torch`, `scikit-learn`, `scipy`, `matplotlib`, `pyyaml`.

---

## Dataset

The dataset file `toy_photon_dataset.npz` should be placed in the **repo root**. It is not included in this repository per the task instructions.

```
X : float32 array of shape (N, 225)   -- calorimeter cell energy deposits
y : float32 array of shape (N,)       -- true photon energy in GeV
```

The calorimeter is modelled as a regular 15×15 grid of cells with unit pitch. Each row of `X` is the flattened (row-major) grid of energy deposits for one event. The dataset used here contains N=200,000 events with true energies in the range [5, 200] GeV.

**If the original dataset is not available**, a synthetic stand-in with the same format and roughly similar characteristics can be generated:

```bash
python generate_dataset.py
```

This produces a 20,000-event `.npz` file using a 2D Gaussian shower model with stochastic energy fluctuations and per-cell Gaussian noise. The synthetic data is only meant for pipeline testing — it does not reproduce the exact statistics of the original dataset.

---

## How to Run

All scripts are run from the repo root. The architecture and all hyperparameters are controlled through `configs/config.yaml`.

**Step 1 — Train**
```bash
python train.py --config configs/config.yaml
```
Trains the model, saves the best and last checkpoints, and saves loss curve plots.

**Step 2 — Evaluate**
```bash
python evaluate.py --config configs/config.yaml
```
Loads the best checkpoint, runs inference on the validation set, computes all metrics, and saves three plots.

**Step 3 — Inference benchmark**
```bash
python infer.py --config configs/config.yaml
```
Measures average inference time per batch and per sample across multiple batch sizes.

**Step 4 — Uncertainty suite**
```bash
python uncertainty_runner.py --config configs/config.yaml
```
Runs all six uncertainty analyses and saves a summary table to `plots/uncertainty/uncertainty_summary.txt`.

**To reproduce all results for all three architectures at 30 epochs:**
```bash
chmod +x run_all.sh
./run_all.sh
```
This patches the config, runs the full pipeline for each architecture in sequence, and saves outputs to separate subdirectories (`plots/mlp/`, `plots/cnn/`, `plots/deepsets/`).

**Switching architectures** — change one line in `config.yaml`:
```yaml
architecture: "mlp"   # options: mlp | cnn | deepsets
```

---

## Models

All three models solve the same regression problem: mapping 225 calorimeter cell energies to a single predicted photon energy. They differ in how they represent and process the input.

### Data Preprocessing

All inputs are standardized per-feature using `sklearn.StandardScaler`. The scaler is fitted on training data only and never applied to the validation set before fitting — this avoids data leakage. The fitted scaler is saved to `checkpoints/scaler.pkl` so inference scripts use exactly the same normalization.

The dataset is split 80/20 (train/validation) using a fixed `random_state=42`.

### MLP

The flat input (225 values) is passed through a stack of fully-connected layers with BatchNorm and Dropout after each hidden layer:

```
Input (225)
  → Linear(225, 256) → BatchNorm1d → ReLU → Dropout(0.2)
  → Linear(256, 128) → BatchNorm1d → ReLU → Dropout(0.2)
  → Linear(128, 64)  → BatchNorm1d → ReLU → Dropout(0.2)
  → Linear(64, 1)
```

Trainable parameters: **99,969**

The MLP treats all 225 cells as independent flat features with no spatial structure. Dropout is included deliberately for two purposes: regularization during training, and Monte Carlo Dropout uncertainty estimation at inference time.

### CNN

The 225-cell vector is reshaped into a 1×15×15 image and processed by three convolutional layers followed by a fully-connected head:

```
Input (1, 15, 15)
  → Conv2d(1→16, 3×3, pad=1) → ReLU → MaxPool2d(2)   [→ 7×7]
  → Conv2d(16→32, 3×3, pad=1) → ReLU → MaxPool2d(2)  [→ 3×3]
  → Conv2d(32→64, 3×3, pad=1) → ReLU
  → Flatten → Linear(576, 128) → ReLU → Dropout(0.2)
            → Linear(128, 64)  → ReLU → Dropout(0.2)
            → Linear(64, 1)
```

Trainable parameters: **105,473**

Convolutional layers exploit the spatial locality of electromagnetic showers — nearby cells are physically correlated, and the CNN encodes this inductive bias explicitly.

### DeepSets

Rather than flattening the grid, DeepSets treats each cell as a point in a set. Each cell gets three input features: its energy and its normalized row/column coordinates. A shared per-cell network (φ) transforms each cell independently, the results are summed across all 225 cells, and a second network (ρ) maps the aggregate to the predicted energy.

```
Per cell: [energy, norm_row, norm_col]  →  φ(64→128→128)  →  cell embedding
Sum across all 225 cells               →  aggregate representation
ρ(64→1)                                →  predicted energy
```

Trainable parameters: **33,411**

The sum-pooling aggregation step is permutation-invariant: the model produces the same output regardless of the order in which cells are presented. This makes DeepSets a natural fit for set-structured inputs like calorimeter readouts, where cell ordering is arbitrary.

### Training Details

All models are trained with:
- Loss: Mean Squared Error
- Optimizer: Adam (lr=1e-3, no weight decay)
- LR scheduler: ReduceLROnPlateau (patience=10, factor=0.5)
- Batch size: 128
- Epochs: 100 (30 for the reproduction run below)
- Seed: 42

Checkpoints are saved at the best validation loss (`{arch}_best.pt`) and at the final epoch (`{arch}_last.pt`). Each checkpoint stores the model weights, the epoch number, the validation loss, and the full config dict — so the model can be reconstructed from the checkpoint alone without needing the config file on disk.

**First-epoch loss spike**: The training loss at epoch 1 is typically much larger than subsequent epochs because the model starts from random weights. `train.py` automatically detects this (if epoch 1 loss > 3× epoch 2 loss) and saves a second loss curve with epoch 1 removed, `loss_curve_clipped.png`, to make the convergence behaviour easier to read.

---

## Results

All results below are from the 30-epoch reproduction run on the original dataset (N=200,000, energy range [5, 200] GeV), evaluated on the 20% validation split (40,000 events).

### Metric Definitions

| Metric | Formula |
|--------|---------|
| MAE | mean\|E_pred − E_true\| |
| RMSE | sqrt(mean((E_pred − E_true)²)) |
| Bias | mean(ΔE/E) × 100% |
| RMS(ΔE/E) | sqrt(mean((ΔE/E)²)) × 100% |
| Std(ΔE/E) | std(ΔE/E) × 100% |
| p68(\|ΔE/E\|) | 68th percentile of \|ΔE/E\| × 100% |

where ΔE/E = (E_pred − E_true) / E_true.

`p68` is the standard physics metric for energy resolution: it is analogous to the 1σ interval of a Gaussian but makes no assumption about the shape of the residual distribution.

### Model Comparison

| Metric | MLP | CNN | DeepSets |
|--------|-----|-----|----------|
| MAE [GeV] | 5.037 | 5.247 | 18.613 |
| RMSE [GeV] | 7.085 | 7.472 | 22.664 |
| Bias [%] | +2.63 | +1.95 | −22.66 |
| RMS(ΔE/E) [%] | 9.09 | 10.11 | 25.17 |
| Std(ΔE/E) [%] | 8.70 | 9.92 | 10.97 |
| p68(\|ΔE/E\|) [%] | 7.98 | 8.06 | 24.12 |
| Parameters | 99,969 | 105,473 | 33,411 |
| Best val loss | 51.07 | 55.88 | 513.78 |

### Plots

Each architecture produces four plots, saved to `plots/{arch}/`:

- `loss_curve.png` — training and validation MSE loss vs epoch
- `loss_curve_clipped.png` — same with epoch 1 removed (where applicable)
- `pred_vs_true.png` — 2D density histogram of predicted vs true energy with ideal diagonal
- `delta_e_hist.png` — histogram of ΔE/E with bias annotated and resolution metrics in a text box
- `resolution_vs_energy.png` — p68(|ΔE/E|) in equal-statistics bins of E_true

---

## Uncertainty Quantification

The uncertainty suite (`uncertainty_runner.py`) runs six analyses on the validation set predictions. All outputs are saved to `plots/{arch}/uncertainty/`.

### Bootstrap Confidence Intervals

Every reported metric is accompanied by a 95% bootstrap confidence interval estimated from 1,000 resamples of the validation set. This gives a sense of how stable the metric estimate is given the finite validation sample size.

MLP results with 95% CIs:

| Metric | Estimate | 95% CI |
|--------|----------|--------|
| Bias [%] | +2.630 | [+2.547, +2.718] |
| RMS(ΔE/E) [%] | +9.087 | [+8.995, +9.167] |
| p68(\|ΔE/E\|) [%] | +7.977 | [+7.895, +8.063] |
| MAE [GeV] | +5.037 | [+4.987, +5.083] |
| RMSE [GeV] | +7.085 | [+7.015, +7.151] |

### Binned Resolution

`p68(|ΔE/E|)` is computed in 10 equal-statistics bins of E_true, with a bootstrap uncertainty bar per bin. This shows how the model's resolution varies across the energy range. Both MLP and CNN show the expected behaviour — better resolution at higher energies where the shower is more contained and the signal-to-noise ratio is higher. The DeepSets model shows poor and roughly flat resolution across all energies, consistent with its large systematic bias.

### Monte Carlo Dropout

With dropout kept active at inference time, 100 stochastic forward passes are run per sample. The standard deviation across passes estimates the model's epistemic (parameter) uncertainty on a per-sample basis.

| Model | Mean σ [GeV] | Std of σ [GeV] |
|-------|-------------|----------------|
| MLP | 6.45 | 3.33 |
| CNN | 8.30 | 5.74 |
| DeepSets | 39.08 | 24.89 |

The MC Dropout uncertainty is notably large relative to the prediction errors for MLP and CNN. This suggests the model's dropout-based uncertainty is not well-calibrated — likely because the dropout rate (0.2) is not tuned for uncertainty estimation. The DeepSets numbers are very large, consistent with the model not having converged.

### Gaussian Fit to ΔE/E

A Gaussian is fitted to the ΔE/E histogram using `scipy.optimize.curve_fit`. The fit parameters and their uncertainties come from the diagonal of the covariance matrix returned by the optimizer.

| Model | Fit μ [%] | μ uncertainty | Fit σ [%] | σ uncertainty |
|-------|-----------|--------------|-----------|--------------|
| MLP | +1.221 | ±0.107 | 7.638 | ±0.107 |
| CNN | +0.232 | ±0.082 | 7.639 | ±0.082 |
| DeepSets | −20.066 | ±0.073 | 6.339 | ±0.073 |

Note that the Gaussian fit μ differs from the bootstrap bias estimate for MLP and CNN. This is because the fit μ characterises the peak of the distribution, while the bootstrap bias is the mean of the full distribution including tails. When the distribution has non-Gaussian tails (as confirmed by the KS test below), these two quantities diverge.

### Deep Ensemble

Five independent models are trained from scratch with different random seeds (42, 7, 13, 99, 2025). The spread of their predictions at inference time gives an ensemble-based epistemic uncertainty estimate.

| Model | Mean σ [GeV] | Std of σ [GeV] |
|-------|-------------|----------------|
| MLP | 0.906 | 0.459 |
| CNN | 2.230 | 1.065 |
| DeepSets | 11.677 | 7.885 |

Ensemble uncertainty is considerably lower than MC Dropout uncertainty for MLP and CNN. This is expected: the ensemble members have converged to similar solutions in weight space, so their predictions agree closely. MC Dropout explores a much wider range of effective model configurations per forward pass.

### KS Test and Shapiro-Wilk Test

Both tests assess whether the ΔE/E residuals follow a Gaussian distribution. The diagnostic plot shows the empirical CDF vs the theoretical Gaussian CDF, and a QQ plot.

| Model | KS stat | KS p-value | SW stat | SW p-value |
|-------|---------|-----------|---------|-----------|
| MLP | 0.049 | 0.000 | 0.971 | 0.000 |
| CNN | 0.073 | 0.000 | 0.919 | 0.000 |
| DeepSets | 0.137 | 0.000 | 0.843 | 0.000 |

All three models produce non-Gaussian residual distributions (p=0 for both tests across all architectures). This is physically expected: real calorimeter shower shapes have asymmetric tails from energy leakage into inactive material and noise-floor effects. A consequence is that `p68(|ΔE/E|)` is a more robust resolution metric than the Gaussian fit σ, since it makes no distributional assumption.

---

## Inference Performance

Measured on an NVIDIA GPU, batch size swept from 1 to 1024, with 10 warm-up batches before timing and 200 timed repetitions.

### MLP

| Batch size | Time/batch [ms] | Time/sample [µs] |
|------------|----------------|-----------------|
| 1 | 0.635 | 634.9 |
| 32 | 0.831 | 26.0 |
| 256 | 0.836 | 3.3 |
| 1024 | 0.972 | 0.9 |

### CNN

| Batch size | Time/batch [ms] | Time/sample [µs] |
|------------|----------------|-----------------|
| 1 | 0.656 | 656.4 |
| 32 | 0.628 | 19.6 |
| 256 | 0.925 | 3.6 |
| 1024 | 1.365 | 1.3 |

### DeepSets

| Batch size | Time/batch [ms] | Time/sample [µs] |
|------------|----------------|-----------------|
| 1 | 0.564 | 564.0 |
| 32 | 0.511 | 16.0 |
| 256 | 2.194 | 8.6 |
| 1024 | 8.483 | 8.3 |

The MLP and CNN scale well to large batch sizes — per-sample latency drops by roughly 700× going from batch size 1 to 1024. DeepSets does not: its per-sample time stays high at large batches because the per-cell φ network processes 225 tokens per event, making the effective computation 225× larger per sample than for the flat models. At batch size 1024, DeepSets takes ~8.3 µs/sample vs ~0.9 µs for MLP.

---

## Discussion

### Model Performance

The MLP achieves the best overall performance across all metrics at 30 epochs, with p68 = 7.98% and a bias of +2.63%. The CNN performs comparably (p68 = 8.06%, bias = +1.95%) with slightly larger spread metrics, likely because the 15×15 grid is too small for convolutions to provide a strong advantage over flat features — there are only two pooling steps before the spatial resolution is reduced to 3×3. With more epochs both models would likely continue to improve.

The DeepSets model performs poorly at 30 epochs, with a large negative bias of −22.66% and p68 of 24.12%. The training and validation losses are still decreasing steadily at epoch 30, suggesting the model simply needs more training. The per-cell sum-pooling approach is theoretically well-motivated but requires the φ network to learn a representation whose sum captures the total energy — a harder implicit constraint than the MLP's direct regression. Longer training, a larger φ network, or mean-pooling instead of sum-pooling may help.

### Bias

All models show a non-zero bias. For MLP and CNN the bias is a small positive systematic of ~2% and ~2%, meaning predicted energies are slightly higher than true energies on average. This could arise from the asymmetry introduced by ReLU activations (which clip negative values) or from the non-linearity of the energy scale. The Gaussian fit μ (which characterises the peak rather than the mean) is smaller: +1.22% for MLP and +0.23% for CNN, suggesting the bulk of events are nearly unbiased and the mean is pulled by non-Gaussian high-side tails. DeepSets has a large negative bias indicating systematic underestimation, consistent with a model that has not converged.

### Overfitting and Underfitting

For MLP and CNN, the training loss is consistently higher than the validation loss — the reverse of the usual overfitting signature. This is likely due to Dropout being active during training (which adds noise) but inactive during validation. It is a sign of healthy regularization rather than underfitting. Neither model shows the validation loss diverging from training loss as epochs progress, which would indicate overfitting.

DeepSets is clearly underfitting at 30 epochs: both training and validation losses remain very high and are still declining at the final epoch with no sign of convergence.

### Possible Improvements

- **More epochs**: MLP and CNN show steady improvement; running to 100+ epochs would reduce both bias and spread further.
- **Energy-dependent normalization**: Normalizing the target `y` (e.g. predicting log(E) instead of E) can stabilize MSE loss across a wide energy range and reduce the large-loss spike at epoch 1.
- **DeepSets architecture**: Replacing sum-pooling with mean-pooling, adding skip connections in the φ network, or using attention-based pooling could improve convergence speed significantly.
- **Weighted loss**: Weighting the MSE by 1/E² would make the loss function equivalent to minimizing RMS(ΔE/E) directly, aligning the training objective with the physics resolution metric.
- **Data augmentation**: Random rotations and reflections of the 15×15 grid would provide cheap augmentation that the CNN could exploit.
- **Calibration**: All three models show non-zero bias. A simple post-processing calibration curve (predicted vs true energy, fitted with a polynomial) could remove the residual systematic shift.

---

## Conceptual Questions

### 1. Scaling to larger inputs and high-occupancy environments

For larger calorimeter grids (e.g. 50×50 or 100×100), the MLP input dimension grows quadratically and the number of parameters in the first layer becomes prohibitively large. More importantly, a flat MLP loses all spatial structure — it has no mechanism to learn that nearby cells are correlated. CNNs handle larger grids naturally through weight sharing and local receptive fields, but global information only accumulates through many pooling layers, which may discard fine-grained shower structure. DeepSets scales linearly in the number of cells and naturally handles variable-size inputs, but the sum-pooling aggregation becomes a bottleneck at very large occupancy.

High-occupancy environments (many simultaneous particles, pile-up from multiple proton interactions) introduce overlapping showers that violate the single-particle assumption baked into this dataset. A model trained on isolated showers will fail to separate contributions from nearby particles without explicit clustering or a model that conditions on the full event context. This is a fundamental problem that none of the three architectures here address — graph-based approaches that model cell-to-cell interactions explicitly are better suited.

### 2. Graph-based and Transformer-based models vs this baseline

The MLP treats the 225-cell input as an unordered flat vector with no geometric awareness. The cell at index 0 has no special relationship to the cell at index 1 in the model's representational framework, even though they are adjacent on the grid.

A graph neural network would represent each cell as a node and connect nodes by edges encoding spatial proximity (e.g. within a radius of 2 cells). Message passing allows each node to aggregate information from its neighbours, so the model explicitly reasons about local shower shape. This also handles irregular detector geometries and variable-occupancy inputs naturally, since the graph topology can be event-specific.

A Transformer treats each cell as a token and computes pairwise attention weights between all 225 cells. This allows the model to learn long-range correlations across the shower without the locality constraint of convolutions. Positional encodings (row, column coordinates) inject geometry into the otherwise permutation-invariant attention mechanism. Unlike the MLP or CNN, the Transformer can in principle learn which cells to attend to conditionally on the shower shape, which is physically meaningful — the core of the shower carries more information than the periphery, and a well-trained attention mechanism should reflect this.

Both graph and Transformer approaches preserve permutation equivariance in a principled way, which the MLP with a fixed flattening order does not.

### 3. Reducing latency under real-time constraints

In the LHC trigger context, inference must run within microseconds per event to keep pace with the 40 MHz bunch crossing rate. Strategies include:

**Model compression**: Post-training quantization (INT8 or even INT4 weights) reduces memory bandwidth and enables faster arithmetic on supported hardware. Structured pruning removes entire neurons or filters whose contributions are small, reducing the effective model size without changing the architecture. Knowledge distillation trains a smaller student model to reproduce the outputs of the larger trained model.

**Deployment tooling**: Exporting to ONNX and compiling with TensorRT (NVIDIA) or OpenVINO (Intel) provides hardware-specific kernel fusion and operator optimization that can reduce latency by 2–5× relative to PyTorch eager mode. TorchScript compilation removes Python overhead from the inference path.

**FPGA inference**: For the strictest latency requirements, the hls4ml library converts trained PyTorch/Keras models to synthesizable HLS (High-Level Synthesis) code that runs on FPGAs. This is the actual deployment pathway used in the CMS and ATLAS Level-1 trigger systems, where latency budgets are on the order of a few hundred nanoseconds. The MLP architecture implemented here is well-suited to FPGA deployment — small parameter count, no dynamic control flow, fixed input size.

**Architectural choices**: Smaller hidden dimensions, fewer layers, and avoiding operations that are expensive on GPU (e.g. large matrix transposes, dynamic shapes) all reduce latency. For this specific problem, the MLP's sub-1-µs per-sample time at large batch sizes already approaches a regime compatible with online reconstruction, making it the most deployment-ready of the three architectures evaluated here.
