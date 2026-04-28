import pandas as pd

from features import build_features, get_feature_columns
from model import train_model
from pipeline import simulate_future
from schedule_maker import build_calendar

df = pd.read_csv('data/raw_data.csv')

df_features = build_features(df)
feature_cols = get_feature_columns(df_features)

model = train_model(df_features, feature_cols)

future_schedule = build_calendar(
    start_date='2026-04-01',
    end_date='2026-05-01'
)

df_sim, future_results = simulate_future(df, model, future_schedule, noise_std=10)
print(future_results.head())
