from pathlib import Path

# Base Directory Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Grid Dimensions
NZ = 64
NY = 1024

# Physical Parameters
GAMMA = 0.05
TAU = 0.01
RA_DEFAULT = 1e8
PR_DEFAULT = 1.0

# Ensure directories exist upon initialization
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)
