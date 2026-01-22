from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt


# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)


# Create a boxplot for Median Income
plt.figure(figsize=(14, 10))
plt.boxplot(df['MedInc'], vert=False, patch_artist=True)
plt.title('Distribution of Median Income (California Housing)')
plt.xlabel('Median Income')
plt.grid(True, linestyle='--', alpha=0.7)

# Save the boxplot to the figs folder
output_path = 'figs/boxplot.png'
plt.savefig(output_path)
print(f"Success! Plot saved to {output_path}")