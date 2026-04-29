import pandas as pd

from features import build_features, get_feature_columns
from model import train_model
from reports import save_rankings, save_averages
from schedule_maker import build_calendar
from pipeline import run_simulations, compute_expected_cutoff, compute_expected_player_points

df = pd.read_csv('data/raw_data.csv')
today = df['date'].max()
inactive_players = [
    'Дядя Витя',
    'Кесадилия',
    'E_ishutkin',
    'yakushevass'
]

save_rankings(df, today)
save_averages(df)

df_features = build_features(df)
feature_cols = get_feature_columns(df_features)

model = train_model(df_features, feature_cols)

future_schedule = build_calendar(
    start_date='2026-04-01',
    end_date='2026-05-01',
    overrides= {
        pd.Timestamp('2026-05-01'): 'bounty'
    }
)

cutoff_history, player_history = run_simulations(df, model, future_schedule, n_sim=10, noise_std=15, inactive_players=inactive_players)
expected_cutoff = compute_expected_cutoff(cutoff_history)
expected_players = compute_expected_player_points(player_history)

print('Expected cutoff:', expected_cutoff)
print(expected_players.head(20))
