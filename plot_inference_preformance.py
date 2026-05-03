from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import io

PROJECT_DIR = Path(__file__).parent
data_dir = PROJECT_DIR / "metadata"
data_file = data_dir / "data.csv"


df = pd.read_csv(data_file)

# 2. Pre-processing
# Since your time values are Unix timestamps, we calculate the relative 
# duration by subtracting the minimum timestamp in the set.
df['time_rel'] = df['time'] - df['time'].min()

# Group by method and size to calculate Mean and Standard Deviation
stats = df.groupby(['method_name', 'size'])['time_rel'].agg(['mean', 'std']).reset_index()

# 3. Plotting
plt.figure(figsize=(12, 7))

for method in stats['method_name'].unique():
    subset = stats[stats['method_name'] == method].sort_values('size')
    
    # 'yerr' creates the "candle stick" effect for Standard Deviation
    plt.errorbar(subset['size'], subset['mean'], yerr=subset['std'], 
                 label=method, marker='o', capsize=5, linestyle='--', linewidth=1.5)

# Formatting the chart
plt.xscale('log')  # Best for visualizing sizes like 100, 1000, 10000
plt.title('Execution Time by Method and Dataset Size (Mean ± STD)', fontsize=14)
plt.xlabel('Dataset Size (Log Scale)', fontsize=12)
plt.ylabel('Relative Time (Seconds)', fontsize=12)
plt.legend(title="ML Methods")
plt.grid(True, which="both", ls="-", alpha=0.3)

plt.tight_layout()
plt.savefig("plots/inference_times.png")