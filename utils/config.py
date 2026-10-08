"""Portable defaults; relative environment paths are resolved against repository root."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def configured_path(name, default):
    value = Path(os.environ.get(name, str(default))).expanduser()
    return value if value.is_absolute() else ROOT / value

IMAGE_DIR = configured_path('LAB2_IMAGE_DIR', ROOT / 'dataset/images')
RESULT_DIR = configured_path('LAB2_RESULT_DIR', ROOT / 'results')
_local_data = ROOT / 'dataset/dota'
DATA_DIR = configured_path('LAB2_DATA_DIR', _local_data if _local_data.exists() else ROOT.parent / 'train')
SAM_CHECKPOINT = configured_path('LAB2_SAM_CHECKPOINT', ROOT / 'asset/sam_vit_b_01ec64.pth')
APPEARANCE_WEIGHTS = configured_path('LAB2_APPEARANCE_WEIGHTS', ROOT / 'asset/resnet18-f37072fd.pth')
CLASSIFIER_MODEL = configured_path('LAB2_CLASSIFIER_MODEL', ROOT / 'asset/dota_classifier/model.joblib')
