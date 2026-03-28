#!/usr/bin/env python3
"""
Extract AI-generated text from batch CSV files and append to the main dataset.
This script:
1. Reads content from batch CSV files in AI_text/AI_text/gpt/
2. Extracts features using data_pipe_extract_features.py
3. Appends to data.csv with -AI label (no -schema flag)
"""

import csv
import subprocess
import os
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).parent
AI_DATA_DIR = PROJECT_DIR / "AI_text" / "AI_text" / "gpt"
DATA_CSV = PROJECT_DIR / "data.csv"
FEATURE_SCRIPT = PROJECT_DIR / "data_pipe_extract_features.py"

# Batch CSV files to process
BATCH_FILES = [
    "batch1.csv",
    "batch2.csv", 
    "batch3.csv"
]

def extract_text_from_csv(csv_file):
    """Extract content column from a CSV file."""
    texts = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'content' in row and row['content']:
                    texts.append(row['content'])
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
    return texts

def append_ai_data_to_dataset():
    """Extract AI data from batch files and append to dataset."""
    all_texts = []
    
    print("Extracting text from batch CSV files...")
    for batch_file in BATCH_FILES:
        batch_path = AI_DATA_DIR / batch_file
        if batch_path.exists():
            print(f"  Processing {batch_file}...")
            texts = extract_text_from_csv(batch_path)
            all_texts.extend(texts)
            print(f"    Extracted {len(texts)} texts")
        else:
            print(f"  Warning: {batch_file} not found at {batch_path}")
    
    if not all_texts:
        print("No AI texts extracted!")
        return False
    
    print(f"\nTotal texts extracted: {len(all_texts)}")
    
    # Combine all texts with newlines to separate them
    combined_text = "\n\n".join(all_texts)
    
    print(f"\nAppending {len(all_texts)} AI-generated texts to {DATA_CSV.name}...")
    print("Running feature extraction")
    
    try:
        # Pipe the combined text through the feature extraction script
        # Using -pipe, -AI flags, and -S for chunk size
        # NO -schema flag as per instructions
        cmd = [
            "python3",
            str(FEATURE_SCRIPT),
            "-AI",
            "-pipe",
            "-S", "500"  # Chunk size of 500 tokens
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=PROJECT_DIR,
            text=True
        )
        
        stdout, stderr = proc.communicate(input=combined_text)
        
        if stderr:
            print(f"Errors during feature extraction:\n{stderr}")
        
        if proc.returncode != 0:
            print(f"Feature extraction failed with return code {proc.returncode}")
            return False
        
        # Append the output to data.csv (skip header from pipe output)
        with open(DATA_CSV, 'a') as f:
            f.write(stdout)
        
        print(f"Successfully appended {len(all_texts)} AI samples to {DATA_CSV.name}")
        return True
        
    except Exception as e:
        print(f"Error during appending: {e}")
        return False

if __name__ == "__main__":
    os.chdir(PROJECT_DIR)
    success = append_ai_data_to_dataset()
    exit(0 if success else 1)
