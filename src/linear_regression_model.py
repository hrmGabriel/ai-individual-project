from dataclasses import dataclass

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    model: RegressorMixin
    y_pred: np.ndarray
    scaler: StandardScaler | None = None


def train_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> ModelResult:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    return ModelResult(model=model, y_pred=y_pred, scaler=scaler)
