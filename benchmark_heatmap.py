import time
import pandas as pd
import numpy as np

def original_method(df, df_normalized, best_values):
    start = time.time()
    for _ in range(10):
        count = 0
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                metric_value = df.iloc[i, j]
                is_best = best_values and np.isclose(metric_value, best_values[df.columns[j]])
                color = "black" if df_normalized.iloc[i, j] > 0.5 else "white"
                count += 1
    return time.time() - start

def optimized_method(df, df_normalized, best_values):
    start = time.time()
    for _ in range(10):
        count = 0
        df_values = df.values
        df_norm_values = df_normalized.values
        columns = df.columns
        for i, row in enumerate(df_values):
            for j, metric_value in enumerate(row):
                is_best = best_values and np.isclose(metric_value, best_values[columns[j]])
                color = "black" if df_norm_values[i, j] > 0.5 else "white"
                count += 1
    return time.time() - start

df = pd.DataFrame(np.random.rand(50, 20), columns=[f'col_{i}' for i in range(20)])
df_normalized = pd.DataFrame(np.random.rand(50, 20), columns=[f'col_{i}' for i in range(20)])
best_values = {f'col_{i}': np.random.rand() for i in range(20)}

print(f"Original: {original_method(df, df_normalized, best_values):.4f}s")
print(f"Optimized: {optimized_method(df, df_normalized, best_values):.4f}s")
