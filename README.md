# MGraph: Adaptive Multi-Graph Neural Network for Traffic Flow Forecasting

This repository contains the implementation of **MGraph**, a novel adaptive multi-graph neural network for dynamic spatial-temporal traffic flow forecasting, as presented in the paper *"MGraph: Adaptive Multi-Graph Neural Network for Dynamic Spatial-Temporal Traffic Flow Forecasting"* , submitted to IEEE Transactions on Intelligent Transportation Systems (T-ITS).

MGraph leverages an adaptive adjacency matrix and a weekly regularity scheme to model evolving spatial dependencies and temporal periodicity in traffic flow data. The model achieves state-of-the-art performance on datasets like PeMS03, PeMS04, PeMS07, PeMS08, and METR-LA, with significant accuracy (e.g., 8.7% MAE reduction on PeMS08) and computational efficiency (e.g., 199.6% runtime reduction on PeMS08).

## Repository Structure

- `config/`: Configuration files (e.g., `PeMS03.conf...`, `MGraph.conf`) for datasets and model hyperparameters.
- `execute/`: Training scripts (`base_trainer.py`, `mgraph_trainer.py`) for model training and evaluation.
- `util/`: Utility scripts for data loading, logging (`logger.py`), and model initialization.
- `model/`: MGraph model implementation (`model.py`).
- `dataset/`: Contains PeMS03 and PeMS04 data (`PEMS03.npz`, `PEMS03.txt`, `PEMS03.csv`, `...`).
- `checkpoints/`: Stores trained model weights (e.g., `mgraph_PeMS08_seed1.pth`).
- `log/`: Stores training logs (e.g., `MGraph_PeMS08_seed1.txt`).
- `points/`: Stores prediction outputs (e.g., `points_MGraph_PeMS08_seed1.npz`).
- `scripts/`: (Optional) Analysis script for parsing results.

## Prerequisites

- **Python**: 3.8 or higher

- **PyTorch**: 1.9 or higher

- **CUDA**: 11.6 or higher (for GPU training, e.g., NVIDIA RTX 4090)

- **Dependencies**: Install required packages:

- **Hardware**: GPU recommended for faster training.

## Setup

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/JABUBROWN/MGraph.git
   cd MGraph/
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Datasets**:

   - PeMS03 and PeMS04 are included in `datasets/` .

   - Download PeMS07, PeMS08, and METR-LA Google Drive (\[https://drive.google.com/drive/folders/1UtJ017rrQrvdhND8t1ewLBNBY83O0fgv?usp=sharing\]).

   - Place data files in `dataset/` with the following structure:

     ```
     dataset/
     ├── PeMS03
     │   ├── PEMS03.npz
     │   ├── PEMS03.txt
     │   └── PEMS03.csv
     ├── PeMS04
     │   ├── PEMS04.npz
     │   ├── PEMS04.txt
     │   └── PEMS04.csv
     ...
     ```

   - **Required Files**:

     - PeMS datasets:
       - `.npz`: Preprocessed traffic flow data (e.g., vehicle counts over time steps).
       - `.csv`: Static adjacency data.
       - （PeMS03)-（`.txt`: Sensor ID mappings (e.g., mapping `313344` to index `0`).
     - METR-LA: `.h5` (traffic flow data).

   - Experimental data is available at IEEE DataPort (DOI: 10.21227/matk-gf81).

## Usage

1. **Train the Model**: Run `main.py` with a dataset and seed:

   ```bash
   python main.py --data_config_file PeMS03.conf --model_config_file MGraph.conf --seed 1 --device 0
   ```

   - `--data_config_file`: Dataset config (e.g., `PeMS03.conf`).
   - `--model_config_file`: Model config (e.g., `MGraph.conf`).
   - `--seed`: Random seed (1 to 10 for reproducibility).
   - `--device`: GPU ID (e.g., `0`) or `cpu`.
   - **Outputs**:
     - Log file: `log/MGraph_PeMS03_seed1.txt` (training and evaluation metrics, e.g., MAE, RMSE, MAPE).
     - Checkpoint: `checkpoints/mgraph_PeMS03_seed1.pth` (best model weights).
     - Predictions: `points/points_MGraph_PeMS03_seed1.npz` (time steps, sensor indices, ground truth, and predicted traffic flow values).

2. **Test the Model**: Run with `--stage test`:

   ```bash
   python main.py --data_config_file PeMS03.conf --model_config_file MGraph.conf --seed 1 --device 0 --stage test
   ```

   - Requires a trained checkpoint (e.g., `checkpoints/mgraph_PeMS03_seed1.pth`).
   - Evaluates on the test set, logging MAE, RMSE, and MAPE to `log/MGraph_PeMS03_seed1.txt` and saving predictions to `points/points_MGraph_PeMS03_seed1.npz`.
   - Predictions include per-sensor, per-time-step traffic flow forecasts (e.g., vehicle counts) for analysis or visualization.

3. **Analyze Results**:

   - Log files (e.g., `log/MGraph_PeMS08_seed1.txt`) contain per-epoch training/validation losses and test metrics.
   - Prediction files (`.npz`) include `time` (time steps), `space` (sensor indices), `label` (ground truth), and `predict` (forecasted) arrays for per-sensor traffic flow analysis.
   - Example: Load predictions with `np.load('points/points_MGraph_PeMS03_seed1.npz')` to compute custom metrics or plot time series.

## Datasets

The MGraph model is evaluated on five traffic flow datasets:

- **PeMS03**: 358 sensors, 26,208 time steps.
- **PeMS04**: 307 sensors, 16,992 time steps.
- **PeMS07**: 883 sensors, 28,224 time steps.
- **PeMS08**: 170 sensors, 17,856 time steps.
- **METR-LA**: 207 sensors, 34,272 time steps.

## Ablation Study

The ablation study (Section IV of the paper) evaluates two variants:

- **w/o Weekly Regularity**: Removes the weekly regularity scheme.
- **w/o Adaptive Adjacency**: Uses a static adjacency matrix.

Results are detailed in the paper (Table II) and DataPort (`Final_Experiments/Results/Ablation`, DOI: 10.21227/matk-gf81).

## Reproducibility

- **Random Seeds**: Experiments use seeds 1–10 for robust results (set via `--seed`).
- **Configs**: Use `config/PeMS*.conf` and `MGraph.conf` for dataset and model settings.
- **Logs**: Training and evaluation metrics are logged to `log/` (e.g., `MGraph_PeMS08_seed1.txt`).
- **Code Fixes**: The `log/` directory is automatically created (see `util/logger.py`). The `--stage test` option now correctly runs testing only.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Contact

For issues or questions, please open a GitHub issue.
