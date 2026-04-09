# Triton Surrogate Modeling

This repository contains a **PyTorch reimplementation** of the Triton surrogate model pipeline for flood inundation modeling (Sugar Creek case study).  
The workflow supports **data preprocessing, hyperparameter tuning, training, and prediction/export**.

## Repository Structure

```text
├── data_preprocessing/ # prepare data from Triton output
├── data_loader.py # Loads rasters (targets) + tabular CSV (features), handles scaling
├── tuning.py # Runs Optuna-based hyperparameter tuning
├── train.py # Trains model with best params (or defaults), saves checkpoint
├── predict.py # Runs inference, exports GeoTIFFs, mosaics, metrics, histogram
├── tritonlite_sugar_creek.cfg # Config file with paths, columns, settings
└── artifacts/ # Saved configs, checkpoints, outputs
```

## ⚙️ Requirements

- Python 3.9+
- PyTorch >= 2.0  
- Optuna (for hyperparameter tuning)  
- scikit-learn  
- rasterio  
- matplotlib  
- numpy, pandas, PyYAML, configparser  

Install with:

```bash
pip install torch optuna scikit-learn rasterio matplotlib numpy pandas pyyaml
```

## Data Preprocessing
The data_preprocessing directory contains all scripts and configuration files required to transform raw hydrologic and depth-related data into machine-learning–ready inputs and outputs for the Triton / Triton-Lite surrogate modeling workflow. The preprocessing pipeline is organized into two main stages:

- HYG processing: preparation of input features (X)

- Depth processing: preparation of target variables (Y)

1. 01_hyg_processing: Input Feature Preparation (X)

This folder handles preprocessing of HYG time-series data, which serve as model inputs. Each HYG file corresponds to a hydrologic event and contains temporally ordered measurements that are converted into uniform, high-resolution input sequences.

- 01_extract_hyg_from_zipfiles.py: Extracts raw HYG files from ZIP archives.

- 02_convert_hyg_3hrs_to_30mins_t.py: Converts HYG time series from 3-hour intervals to 30-minute intervals.

- convert_hyg_3hrs_to_30mins.cfg: Configuration file controlling temporal conversion parameters.

Output:
Cleaned and temporally standardized HYG time series used as model inputs (X).

2. 02_depth_processing: Target Variable Preparation (Y)

This folder prepares depth-related outputs used as supervised learning targets. Processing focuses on converting event-level depth simulations into spatially organized, block-level representations suitable for regression or surrogate modeling.

Contents:

- 01_generate_netcdf.py
Generates NetCDF files from raw depth simulation outputs.

- 02_conasauga_block_creation.py
Creates block definitions for the Conasauga (e.g., Sugar Creek) watershed.

- 03_netcdf_to_MOM_raster.py
Converts NetCDF depth data into raster format.

- 04_block_selection_to_csv_export.py
Aggregates raster data into block-level CSV outputs.

- 05_extract_raster_from_netcdf.py
Extracts raster layers directly from NetCDF files.

- 06_extract_tiff_from_rasters.py
Exports GeoTIFF files from raster datasets.

- pair_loc_watershed.py
Maps spatial locations to their corresponding watershed.

- directories.cfg
Centralized configuration for input/output directory management.

Output:
Event-aligned, block-level depth data used as model targets (Y).

## Workflow

1. Prepare config file

All paths and settings are defined in Triton_Lite_Ornl/tritonlite_sugar_creek.cfg.
This includes:

- Paths to raster data (base_dir)

- Path to tabular CSV (hyg_dir)

- Output directories

- Number of blocks per set

- Columns to keep (features)

Make sure your data directory structure matches the expected layout.

2. Hyperparameter Tuning

Run random search with Optuna pruning:

```bash
python tuning.py
```

- Uses a 10% validation split

- Runs 20 trials (default), each with early stopping

- Saves best config to artifacts/best_config.yaml

3. Training

Train with tuned params (if available) or defaults:

```bash
python train.py
```

4. Prediction & Export

Run inference on test set:

```bash
python predict.py
```

- Loads checkpoint (artifacts/best.pt)

- Predicts flattened targets

- Splits/reshapes predictions into (bands, H, W) blocks

- Exports GeoTIFFs for each block

- Builds mosaics

- Computes per-pixel maxima (MOM rasters)

- Subtracts GT vs. predictions, writes diff raster

- Prints metrics (F2, CSI, RMSE)

- Saves histogram plot to result_dir

## Outputs

1. artifacts/

    - best_config.yaml → tuned hyperparams

    - best.pt → trained model weights

2. base_dir_tritonlite/ (from config)

    - Block-level predicted GeoTIFFs

    - Mosaic + MOM rasters

3. result_dir/ (from config)

    - diff_pred_vs_gt.tif → difference raster

    - sugar_creek_histogram.png → histogram plot of differences


## Contributing

We welcome contributions!  

1. Fork the repository.  
2. Create a feature branch:  
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. Commit your changes and open a Merge Request (MR).  

## License

TRITON-Lite is released under the **3-Clause BSD License**. See the [LICENSE](LICENSE) file for full terms and conditions.

External third-party libraries included or referenced by TRITON retain their own respective licenses, which are provided in the **licenses** subdirectory.



## Acknowledgments

Development of TRITON-Lite is supported by the U.S. Air Force Numerical Weather Modeling Program. TRITON used resources of the Oak Ridge Leadership Computing Facility at Oak Ridge National Laboratory, a U.S. Department of Energy user facility. Development is led by Oak Ridge National Laboratory, and Tennessee Technological University (Cookeville, TN).

## Contact

Questions, bug reports, or feature requests:
- Open a GitLab issue: <https://code.ornl.gov/hydro/triton-lite/-/issues>
- Email: [Haoran Niu](mailto:niuh@ornl.gov), [Sudershan Gangrade](mailto:gangrades@ornl.gov)
