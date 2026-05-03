from pathlib import Path
import time
import os
import numpy as np
PROJECT_DIR = Path(__file__).parent
output_dir = PROJECT_DIR / "metadata"
out_file = output_dir / "data.csv"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    with open(out_file,'a') as f:
        print("method_name,size,time", file=f)



def time_and_recordFN(predict_fn,method_name):
    X = np.random.rand(10000, 41)

    # Measure inference times
    sample_sizes = [100, 500, 1000, 5000, 10000]
    for size in sample_sizes:
        X_test = np.random.rand(size, 41)
        start = time.time()
        predict_fn(X_test)
        end = time.perf_counter()
        with open(out_file,'a') as f:
            print(f"{method_name},{size},{start-end}", file=f)


