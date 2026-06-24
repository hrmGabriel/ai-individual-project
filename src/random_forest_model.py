import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.linear_regression_model import ModelResult

N_ESTIMATORS = 100


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> ModelResult:
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return ModelResult(model=model, y_pred=y_pred)
