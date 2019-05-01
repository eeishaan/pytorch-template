import os
from pathlib import Path

# project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODEL_DIR = Path(os.path.abspath(PROJECT_ROOT / '../saved_models'))
RESULT_DIR = Path(os.path.abspath(PROJECT_ROOT / '../results'))
PARAM_DIR = PROJECT_ROOT / 'params'
LOG_DIR = PROJECT_ROOT / 'log'

# data file paths
DATA_ROOT_FOLDER = PROJECT_ROOT / 'data'
