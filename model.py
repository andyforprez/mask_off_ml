from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib

def train_model(df, feature_cols):
    X = df[feature_cols]
    y = df['points']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05)

    model.fit(X_train, y_train)
    model.feature_names_ = feature_cols

    print('Train score:', model.score(X_train, y_train))
    print('Val score:', model.score(X_val, y_val))

    return model

def save_model(model, path='model.pkl'):
    joblib.dump(model, path)

def load_model(path='model.pkl'):
    return joblib.load(path)