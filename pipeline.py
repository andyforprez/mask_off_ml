import pandas as pd
from features import build_features
from predict import predict_day
import numpy as np

def apply_inactive_players(df_new, inactive_players):
    if not inactive_players:
        return df_new
    df_new.loc[df_new['player_id'].isin(inactive_players), 'points'] = 0
    return df_new

def simulate_one_run(df_hist, model, schedule, noise_std, inactive_players=None):
    df_sim = df_hist.copy()

    for date, t_type in schedule:
        players = df_sim['player_id'].unique()

        new_rows = pd.DataFrame({
            'player_id': players,
            'date': date,
            'tournament_type': t_type,
            'points': 0
        })

        df_temp = pd.concat([df_sim, new_rows], ignore_index=True)
        df_temp = build_features(df_temp)

        df_new = df_temp[df_temp['date'] == date]

        preds = predict_day(model, df_new, model.feature_names_, noise_std)

        new_rows = new_rows.merge(preds, on='player_id')
        new_rows['points'] = new_rows['predicted_points']
        new_rows.drop(columns=['predicted_points'], inplace=True)

        new_rows = apply_inactive_players(new_rows, inactive_players)

        df_sim = pd.concat([df_sim, new_rows], ignore_index=True)
    return df_sim

def run_simulations(df_hist, model, schedule, n_sim=1000, noise_std=10, inactive_players=None):
    cutoff_history = []
    player_history = []

    for i in range(n_sim):
        df_sim = simulate_one_run(df_hist, model, schedule, noise_std, inactive_players)
        standings = df_sim.groupby('player_id')['points'].sum().sort_values(ascending=False)

        cutoff = standings.iloc[17] if len(standings) >= 18 else np.nan

        cutoff_history.append(cutoff)
        player_history.append(standings)

    return cutoff_history, player_history

def compute_expected_cutoff(cutoff_history):
    return sum(cutoff_history) / len(cutoff_history)

def compute_expected_player_points(player_history):
    df = pd.concat(player_history, axis=1).fillna(0)
    return df.mean(axis=1).sort_values(ascending=False)