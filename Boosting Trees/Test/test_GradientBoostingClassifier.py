import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from model.GradientBoostingClassifier import GradientBoostingClassifier

def test_generated_csv_classification():
    csv_path = "tests/small_test_class.csv"
    print("Reading file:", csv_path)

    df = pd.read_csv(csv_path)
    target_col = df.columns[-1]
    print("Detected target column:", target_col)

    X = df.drop(target_col, axis=1).values
    y = df[target_col].values.astype(int)

    print("Target distribution:", np.unique(y, return_counts=True))

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)

    preds = model.predict(X)
    acc = (preds == y).mean()

    print("Classification accuracy from CSV data:", acc)
    assert acc > 0.8