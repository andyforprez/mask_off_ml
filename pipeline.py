import pandas as pd
from features import build_features
from predict import predict_day

def simulate_future(df_hist, model, schedule, noise_std=None):
    df_sim = df_hist.copy()

    all_results = []

    for date, t_type in schedule:
        players = df_sim['player_id'].unique()

        new_rows = pd.DataFrame({
            'player_id': players,
            'date': date,
            'tournament_type': t_type,
            'points': 0
        })
        new_rows = new_rows.drop_duplicates(subset=['player_id', 'date'])

        df_temp = pd.concat([df_sim, new_rows], ignore_index=True)
        df_temp = build_features(df_temp)
        feature_cols = model.feature_names_

        df_new = df_temp[df_temp['date'] == date]

        preds = predict_day(model, df_new, feature_cols, noise_std)

        new_rows = new_rows.merge(preds, on='player_id')
        new_rows['points'] = new_rows['predicted_points']
        new_rows.drop(columns=['predicted_points'], inplace=True)

        all_results.append(new_rows)
        df_sim = pd.concat([df_sim, new_rows], ignore_index=True).drop_duplicates(subset=['player_id', 'date'],
                                                                                  keep='last')

    return df_sim, pd.concat(all_results)