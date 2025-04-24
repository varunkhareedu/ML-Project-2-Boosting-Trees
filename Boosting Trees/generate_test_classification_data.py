import pandas as pd
import numpy as np
import os

# Create directory for test files if it doesn't exist
output_dir = "tests"
os.makedirs(output_dir, exist_ok=True)

# --- small_test_class.csv ---
X_small = np.random.randn(10, 3)
y_small = (X_small[:, 0] + X_small[:, 1] > 0).astype(int)

df_small = pd.DataFrame(X_small, columns=["x_0", "x_1", "x_2"])
df_small["y"] = y_small
small_test_path = os.path.join(output_dir, "small_test_class.csv")
df_small.to_csv(small_test_path, index=False)

# --- collinear_class.csv ---
np.random.seed(42)
X_base = np.random.randn(100, 5)

# Add collinear features: X_6 = X_1, X_7 = X_2
X_collinear = np.hstack([
    X_base,
    X_base[:, [0]],  # duplicate of X_1
    X_base[:, [1]],  # duplicate of X_2
    np.random.randn(100, 3)  # random noise columns
])

# Logistic label generation
logits = 2 * X_collinear[:, 0] + 1.5 * X_collinear[:, 9] - 1
probs = 1 / (1 + np.exp(-logits))
y = np.random.binomial(1, probs)

# Create DataFrame and save
columns = [f"X_{i+1}" for i in range(X_collinear.shape[1])]
df = pd.DataFrame(X_collinear, columns=columns)
df["target"] = y
collinear_test_path = os.path.join(output_dir, "collinear_class.csv")
df.to_csv(collinear_test_path, index=False)
small_test_path, collinear_test_path