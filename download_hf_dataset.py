#!/usr/bin/env python3
"""
Download and process Hugging Face dataset using huggingface_hub and pandas.
Downloads from: https://huggingface.co/datasets/gsingh1-py/train
Uses subprocess to pipe text through feature extraction pipeline.
"""

import pandas as pd
import subprocess
from pathlib import Path
import sys

def download_dataset():
    """Download dataset from Hugging Face using huggingface_hub."""
    print("Downloading Hugging Face dataset...")
    print("Reading: hf://datasets/gsingh1-py/train/train.csv\n")
    
    try:
        df = pd.read_csv("hf://datasets/gsingh1-py/train/train.csv")
        print(f"✓ Dataset loaded: {len(df)} rows")
        print(f"  Columns: {list(df.columns)}\n")
        return df
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("Make sure to install: pip install huggingface_hub")
        return None

if __name__ == "__main__":
    df = download_dataset()
    if df is not None:
        print(df)

