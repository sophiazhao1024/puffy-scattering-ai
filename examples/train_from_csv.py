"""Fit saved independent solver labels; no Mathematica or fresh ODE required."""
from pathlib import Path
import numpy as np
from puffy.surrogate import Surrogate, metrics
import json
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    train = np.genfromtxt(root / "results/demo/training.csv", delimiter=",", names=True, encoding="utf-8")
    test = np.genfromtxt(root / "results/demo/holdout.csv", delimiter=",", names=True, encoding="utf-8")
    model = Surrogate.fit(train["b"], train["S"])
    print(json.dumps(metrics(test["S"], model.predict(test["b"])["median"]), indent=2))
    print("Optimizer warnings:", model.warnings)
