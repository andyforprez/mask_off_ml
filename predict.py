import numpy as np
import pandas as pd

def align_features(df, feature_cols):
    df = df.copy()

    for col in feature_cols:
        if col not in df:
            df[col] = 0

    df = df[feature_cols]

    return df


def predict_day(model, df_features, feature_cols=None, noise_std=None):
    if feature_cols is None:
        feature_cols = model.feature_names_

    X = align_features(df_features, feature_cols)

    preds = model.predict(X)

    if noise_std:
        preds += np.random.normal(0, noise_std, size=len(preds))

    df_features['predicted_points'] = np.maximum(preds, 0)

    return df_features[['player_id', 'predicted_points']]