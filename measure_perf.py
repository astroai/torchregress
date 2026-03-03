import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Try to silence matplotlib plots
plt.switch_backend('Agg')

# Original code snippet
def original_method(df, df_normalized, best_values):
    fig, ax = plt.subplots()
    start = time.time()
    for _ in range(10): # Run multiple times for better measurement
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                metric_value = df.iloc[i, j]
                is_best = best_values and np.isclose(metric_value, best_values[df.columns[j]])

                # Format text based on value type
                if isinstance(metric_value, float):
                    text = f"{metric_value:.4f}"
                else:
                    text = str(metric_value)

                # Add bold font for best values
                if is_best:
                    text = f"$\\bf{{{text}}}$"

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="black" if df_normalized.iloc[i, j] > 0.5 else "white",
                )
    end = time.time()
    plt.close(fig)
    return end - start

# Optimized code snippet
def optimized_method(df, df_normalized, best_values):
    fig, ax = plt.subplots()
    start = time.time()
    for _ in range(10): # Run multiple times for better measurement
        df_values = df.values
        df_norm_values = df_normalized.values
        columns = df.columns
        for i, row in enumerate(df_values):
            for j, metric_value in enumerate(row):
                is_best = best_values and np.isclose(metric_value, best_values[columns[j]])

                # Format text based on value type
                if isinstance(metric_value, float):
                    text = f"{metric_value:.4f}"
                else:
                    text = str(metric_value)

                # Add bold font for best values
                if is_best:
                    text = f"$\\bf{{{text}}}$"

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="black" if df_norm_values[i, j] > 0.5 else "white",
                )
    end = time.time()
    plt.close(fig)
    return end - start

# Setup data
n_models = 20
n_metrics = 10

df = pd.DataFrame(np.random.rand(n_models, n_metrics), columns=[f'metric_{i}' for i in range(n_metrics)])
df_normalized = pd.DataFrame(np.random.rand(n_models, n_metrics), columns=[f'metric_{i}' for i in range(n_metrics)])
best_values = {f'metric_{i}': np.random.rand() for i in range(n_metrics)}

orig_time = original_method(df, df_normalized, best_values)
opt_time = optimized_method(df, df_normalized, best_values)

print(f"Original Time: {orig_time:.4f}s")
print(f"Optimized Time: {opt_time:.4f}s")
print(f"Speedup: {orig_time / opt_time:.2f}x")
