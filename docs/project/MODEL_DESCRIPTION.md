# Triton Lite — V3 Blockwise Flood Depth Surrogate Model

## Overview

The V3 model (`BlockwiseFloodMatrixModel`) is a deep learning surrogate that predicts 10 m resolution peak flood depth maps for individual watershed blocks, given a hydrologic event time series and static terrain features. It produces two outputs simultaneously: a continuous depth map and a flood-extent (wet/dry) probability map.

---

## Inputs

### 1. Hydrologic Event Tensor — `480 × 300`

A 2D array representing a single flood event:

- **480** time steps at 30-minute intervals (10 days of simulation)
- **300** spatial locations (streamflow gauge / hydrograph points across the watershed)

Each cell contains a flow value (e.g., discharge in m³/s) at that time and location. This tensor encodes the full temporal dynamics of the flood event.

### 2. Block Scalar Features — 7 variables

Seven scalar descriptors summarizing the geographic and hydrologic properties of the block:

| Feature | Description |
|---|---|
| `centroid_x` | Longitude of block centroid (normalized) |
| `centroid_y` | Latitude of block centroid (normalized) |
| `area` | Block area (m²) |
| `mean_elevation` | Mean DEM elevation within the block (m) |
| `elevation_range` | Max − min elevation within the block (m) |
| `mean_slope` | Mean terrain slope within the block (degrees) |
| `distance_to_outlet` | Distance from block centroid to watershed outlet (m) |

### 3. Block Mask — `80 × 80` binary

A binary grid (1 = valid pixel inside the block, 0 = padding) used to ignore pixels outside the block boundary. Applied as a multiplicative mask on the depth output.

### 4. Static Raster Features — `6 × 80 × 80`

Six spatially-explicit terrain and hydraulic layers, each at 10 m resolution over the 80×80 block grid:

| Channel | Name | Description |
|---|---|---|
| 0 | `dem` | Digital Elevation Model — absolute elevation (m) |
| 1 | `flow_acc` | Flow accumulation — number of upstream drainage cells |
| 2 | `stream_mask` | Binary stream channel mask (1 = stream, 0 = land); resampled using nearest-neighbor |
| 3 | `distance_to_stream` | Euclidean distance from each pixel to the nearest stream channel pixel (m); derived from `stream_mask` via distance transform |
| 4 | `slope` | Terrain slope (degrees) derived from DEM |
| 5 | `relative_elevation` | Elevation of each pixel relative to the nearest stream bed (m); key indicator of flood susceptibility |

All 6 channels are z-score normalized (zero mean, unit variance) per channel using statistics computed from the training set.

---

## Model Architecture

### Temporal Encoder

Compresses the `480 × 300` event tensor into a single 64-dimensional event embedding vector.

```
Input:  batch × 480 timesteps × 300 features
Transpose to: batch × 300 channels × 480 timesteps   (for Conv1D)

Conv1D(300 → 64, kernel=3, pad=1)  →  ReLU  →  BatchNorm1D  →  Dropout
Conv1D(64  → 64, kernel=3, pad=1)  →  ReLU  →  BatchNorm1D  →  Dropout
Conv1D(64  → 64, kernel=3, pad=1)  →  ReLU
AdaptiveAvgPool1D(1)               →  collapses all 480 timesteps

Output: batch × 64   (event embedding)
```

### Block Encoder

Compresses the 7 scalar block features into a 64-dimensional block embedding vector.

```
Input:  batch × 7

Linear(7 → 64)  →  ReLU  →  Dropout
Linear(64 → 64)

Output: batch × 64   (block embedding)
```

### Fusion MLP

Concatenates the event and block embeddings and projects to a spatial seed tensor.

```
Input:  batch × 128   [event_emb (64) || block_emb (64)]

Linear(128 → 128)  →  ReLU  →  Dropout
Linear(128 → 12800)           (= 128 × 10 × 10)

Reshape:  batch × 128 channels × 10 rows × 10 cols

Output: batch × 128 × 10 × 10   (spatial seed)
```

### 3-Stage Decoder

Upsamples the `10×10` spatial seed to `80×80` resolution using three `UpsampleBlock` layers. Each block applies bilinear upsampling followed by a Conv2D to refine spatial features.

Each `UpsampleBlock`:
```
Bilinear Upsample ×2   (no learned parameters)
Conv2D(in → out, kernel=3×3, pad=1)
BatchNorm2D
ReLU
```

| Stage | Input | Output | Spatial size |
|---|---|---|---|
| Seed (from Fusion MLP) | 128 ch | — | 10 × 10 |
| Stage 1 | 128 ch | 64 ch | 20 × 20 |
| Stage 2 | 64 ch | 32 ch | 40 × 40 |
| Stage 3 | 32 ch | 16 ch | 80 × 80 |

### Static Raster Encoder

A lightweight 2-layer CNN that encodes the 6-channel static raster into a 16-channel spatial feature map, preserving the 80×80 resolution throughout.

```
Input:  batch × 6 × 80 × 80

Conv2D(6 → 12, kernel=3×3, pad=1)  →  BatchNorm2D  →  ReLU
Conv2D(12 → 16, kernel=3×3, pad=1) →  BatchNorm2D  →  ReLU

Output: batch × 16 × 80 × 80   (raster embedding)
```

*(The intermediate channel count `mid = max(out_channels, in_channels × 2) = 12`.)*

### Spatial Fusion

The decoder output, block mask, XY coordinate grid, and raster embedding are concatenated channel-wise. No learnable parameters — pure concatenation.

```
Decoder output       16 ch × 80 × 80
Block mask            1 ch × 80 × 80   (binary: 1 inside block)
XY coordinate grid    2 ch × 80 × 80   (normalized −1…+1 row and col position)
Raster embedding     16 ch × 80 × 80   (from Static Raster Encoder)
──────────────────────────────────────
Fused map            35 ch × 80 × 80
```

The XY coordinate grid gives the model explicit spatial position information — useful for learning location-dependent depth patterns within the block.

### Depth Head

Produces the peak flood depth map (meters).

```
Input:  batch × 35 × 80 × 80

Conv2D(35 → 16, kernel=3×3, pad=1)  →  ReLU
Conv2D(16 → 1,  kernel=1×1)
Softplus                              (ensures depth ≥ 0)
× block_mask                          (zeros out padding pixels)

Output: batch × 80 × 80   (depth in meters)
```

`Softplus(x) = ln(1 + eˣ)` — a smooth, everywhere-differentiable approximation to ReLU that enforces non-negativity while allowing strong gradients near zero (shallow/dry pixels).

### Wet Head

Produces a flood-extent probability map.

```
Input:  batch × 35 × 80 × 80

Conv2D(35 → 16, kernel=3×3, pad=1)  →  ReLU
Conv2D(16 → 1,  kernel=1×1)

Output: batch × 80 × 80   (raw logits; apply sigmoid for probabilities)
```

---

## Training Loss

The total loss is a weighted sum of a depth regression loss and an auxiliary wet/dry classification loss:

$$L = L_\text{depth} + 0.3 \times L_\text{wet}$$

### Depth Loss — Depth-Weighted Huber

$$L_\text{depth} = \frac{1}{\sum_i m_i} \sum_i m_i \cdot w_i \cdot H_\delta(\hat{d}_i - d_i)$$

where:
- $m_i$ is the block mask (1 inside block, 0 padding)
- $w_i = 1 + \alpha \cdot \min(d_i, \text{cap})$ is a depth weight that up-weights deeper pixels
- $H_\delta$ is the Huber loss with threshold $\delta$:

$$H_\delta(e) = \begin{cases} \frac{1}{2}e^2 & |e| \leq \delta \\ \delta(|e| - \frac{1}{2}\delta) & |e| > \delta \end{cases}$$

| Hyperparameter | Value (V3) | Role |
|---|---|---|
| `huber_delta` (δ) | 0.25 m | Transition from quadratic to linear penalty |
| `depth_weight_alpha` (α) | 1.0 | Controls how steeply weight increases with depth |
| `depth_weight_cap` | 3.0 m | Maximum depth at which weight is still increased |

### Wet Loss — Binary Cross-Entropy

$$L_\text{wet} = \frac{1}{\sum_i m_i} \sum_i m_i \cdot \text{BCEWithLogits}(z_i;\ \mathbf{1}[d_i \geq \tau_\text{wet}])$$

where $z_i$ is the raw wet logit and $\tau_\text{wet} = 0.05$ m defines the wet/dry boundary.

| Hyperparameter | Value (V3) | Role |
|---|---|---|
| `wet_threshold` (τ_wet) | 0.05 m | Depth threshold for defining a "wet" pixel |
| `aux_wet_loss_weight` | 0.3 | Weight of wet loss relative to depth loss |

---

## Outputs

| Output | Shape | Description |
|---|---|---|
| Depth map | `80 × 80` | Peak flood depth (m) at each 10 m pixel |
| Wet probability map | `80 × 80` | Probability (0–1) that each pixel is flooded |

At inference, a **wet probability threshold** (default 0.05) is applied to the wet probability map to produce a binary flood extent mask. The optimal threshold by F1/CSI is 0.05.

---

## V3 Performance — Test Event D040

| Metric | Value |
|---|---|
| RMSE | 0.061 m |
| MAE | 0.009 m |
| R² | 0.957 |
| Mean Bias (pred − true) | −0.046 m |
| Flood Extent F1 (p_t = 0.05) | 0.964 |
| Flood Extent CSI (p_t = 0.05) | 0.930 |
| % cells within ±0.25 m | 63% |
| % cells within ±1.0 m | 99% |

**Depth-bin RMSE:**

| True Depth Range | RMSE | Mean Bias |
|---|---|---|
| 0 – 0.3 m | 0.198 m | +0.051 m |
| 0.3 – 1 m | 0.331 m | −0.055 m |
| 1 – 2 m | 0.400 m | −0.051 m |
| 2 – 5 m | 0.372 m | −0.131 m |
| > 5 m | 0.641 m | −0.537 m |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Training samples | 213,900 |
| Validation samples | 55,200 |
| Test samples | 6,900 (event D040) |
| Batch size | 32 |
| Learning rate | 1e-3 |
| Dropout | 0.1 |
| Optimizer | Adam |
| Epochs run | 49 |
| Best checkpoint | Epoch 39 (lowest val MSE) |
| Hardware | ORNL Andes (K80 GPU, CUDA 3.7) |
