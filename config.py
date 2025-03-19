from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Data collection parameters
SAMPLING_INTERVAL = 1  # seconds between samples
IDLE_COLLECTION_DURATION = 60  # seconds to collect idle data
LOAD_COLLECTION_DURATION = 30  # seconds to collect data at each load level
LOAD_LEVELS = [20, 40, 60, 80, 100]  # utilization percentages to test
REST_BETWEEN_LOADS = 10  # seconds to rest between load tests

# RTX 2080 Super specifications
RTX_2080_SUPER_SPECS = {
    "name": "NVIDIA GeForce RTX 2080 SUPER",
    "num_sms": 48,
    "cuda_cores": 3072,  # 48 SMs × 64 CUDA Cores per SM
    "memory": 8192,  # 8 GB GDDR6
    "tdp": 250,  # 250W TDP
}

# Model parameters
MODEL_FILENAME = "basic_power_model.pkl"
