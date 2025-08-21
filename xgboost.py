# xgboost_model.py
import numpy as np
from xgboost import XGBRegressor

DEFAULT_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=42,
    tree_method="hist",   # change to "gpu_hist" if you have CUDA
)

class XGBModel:
    def __init__(self, **kwargs):
        p = DEFAULT_PARAMS.copy()
        p.update(kwargs)
        self.model = XGBRegressor(objective="reg:squarederror", **p)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=100,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
