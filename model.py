from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import joblib


def train_model(df, feature_cols):
    X = df[feature_cols]
    y = df['points']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # More regularized params to avoid overfitting on small dataset.
    # subsample + min_samples_leaf act as strong regularizers.
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
    )

    model.fit(X_train, y_train)
    model.feature_names_ = feature_cols

    train_r2  = model.score(X_train, y_train)
    val_r2    = model.score(X_val, y_val)
    val_mae   = mean_absolute_error(y_val, model.predict(X_val))
    train_mae = mean_absolute_error(y_train, model.predict(X_train))

    print(f'  Train R²: {train_r2:.4f}  |  Val R²: {val_r2:.4f}')
    print(f'  Train MAE: {train_mae:.1f}  |  Val MAE: {val_mae:.1f}')

    if train_r2 - val_r2 > 0.3:
        print("  ⚠  Large train/val gap — model may be overfitting. "
              "Predictions will still be used with high noise_std to compensate.")

    return model


def save_model(model, path='model.pkl'):
    joblib.dump(model, path)


def load_model(path='model.pkl'):
    return joblib.load(path)