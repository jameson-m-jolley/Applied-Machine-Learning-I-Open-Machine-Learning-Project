#!/usr/bin/env python3
"""
Download Hugging Face dataset and process through feature extraction pipeline.
Uses huggingface_hub with pandas to read the remote CSV.
Loops through rows and pipes text through the feature extraction script using subprocess.
"""

import pandas as pd
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATA_CSV = PROJECT_DIR / "data.csv"

# AI model columns in the dataset
AI_MODELS = [
    'gemma-2-9b',
    'mistral-7B', 
    'qwen-2-72B',
    'llama-8B',
    'accounts/yi-01-ai/models/yi-large',
    'GPT_4-o'
]

def download_dataset():
    """Download dataset from Hugging Face using pandas + huggingface_hub."""
    print("Downloading Hugging Face dataset...")
    print("Reading: hf://datasets/gsingh1-py/train/train.csv\n")
    
    try:
        df = pd.read_csv("hf://datasets/gsingh1-py/train/train.csv")
        print(f"✓ Dataset loaded: {len(df)} rows")
        print(f"  Columns: {list(df.columns)}\n")
        return df
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Make sure to install: pip install huggingface_hub")
        return None

def process_text_via_pipe(text, label):
    """Pipe text through feature extraction using shell command."""
    try:
        # Use command from readme: cat <file> | python3 data_pipe_extract_features.py -AI -pipe >> data.csv
        cmd = ["python3", "data_pipe_extract_features.py", f"-{label}", "-pipe"]
        
   
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=PROJECT_DIR,
            )

        # 2. Talk to the process (TIMEOUT goes here)
        try:
            stdout_data, stderr_data = process.communicate(input=text, timeout=200)
        except subprocess.TimeoutExpired:
            process.kill()
            # Clean up the pipes after killing
            stdout_data, stderr_data = process.communicate()
            print(f"  [!] Timeout reached for {label}")
            return None

  
        print(stdout_data)
        return stdout_data
    except Exception as e:
        print(f"  Error processing text: {e}")
        return None

def process_dataset_rows(df):
    """Process each row of the dataset through the pipeline."""
    print("Processing dataset rows...")
    print(f"Total rows: {len(df)}\n")
    
    human_count = 0
    ai_count = 0
    
    # Open data.csv in append mode
    with open(DATA_CSV, 'a') as outfile:
        for idx, row in df.iterrows():
            # Process human text
            if pd.notna(row.get('Human_story')) and row['Human_story'].strip():
                output = process_text_via_pipe(row['Human_story'], 'human')
                if output:
                    outfile.write(output)
                    human_count += 1
            
            # Process AI texts from each model
            for model in AI_MODELS:
                if model in row and pd.notna(row[model]) and row[model].strip():
                    output = process_text_via_pipe(row[model], 'AI')
                    if output:
                        outfile.write(output)
                        ai_count += 1
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)} rows...")
    
    return human_count, ai_count

def main():
    """Main execution."""
    print("="*60)
    print("DOWNLOAD & PROCESS HF DATASET")
    print("="*60 + "\n")
    
    # Download dataset
    df = download_dataset()
    if df is None:
        return 1
    
    # Process rows
    print("Extracting features from text and appending to data.csv...\n")
    human_count, ai_count = process_dataset_rows(df)
    
    print(f"\n{'='*60}")
    print(f"✓ SUCCESS: Added {human_count + ai_count} total samples")
    print(f"  - Human: {human_count}")
    print(f"  - AI: {ai_count}")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    import sys
    exit(main())
