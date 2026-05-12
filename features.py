import pandas as pd
import numpy as np

from conversion import TOURNAMENT_METADATA, add_tournament_metadata_columns, normalize_tournament_type

TOURNAMENT_TYPES = set(TOURNAMENT_METADATA)


def preprocess(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['tournament_type'] = df['tournament_type'].map(normalize_tournament_type)
    df = df.sort_values(['player_id', 'date'])
    return df

def add_basic_features(df):
    df = df.copy()

    df['rolling_mean_5'] = df.groupby('player_id')['points'].transform(lambda x: x.shift(1).rolling(5).mean())
    df['rolling_std_5'] = df.groupby('player_id')['points'].transform(lambda x: x.shift(1).rolling(5).std())
    df['expanding_mean'] = df.groupby('player_id')['points'].transform(lambda x: x.shift(1).expanding().mean())
    df['rolling_mean_3'] = df.groupby('player_id')['points'].transform(lambda x: x.shift(1).rolling(3).mean())
    df['rolling_mean_10'] = df.groupby('player_id')['points'].transform(lambda x: x.shift(1).rolling(10).mean())
    df['momentum'] = df['rolling_mean_3'] - df['rolling_mean_10']
    df['games_played'] = df.groupby('player_id').cumcount()
    return df



def add_attendance_features(df):
    df = df.copy()

    df['played'] = 1

    df['attendance_rate_10'] = df.groupby('player_id')['played'].transform(lambda x: x.shift(1).rolling(10).mean())
    return df

def add_time_features(df):
    df = df.copy()
    df['days_since_last'] = df.groupby('player_id')['date'].diff().dt.days
    df['days_since_last'] = df['days_since_last'].fillna(999)
    return df

def add_tournament_features(df):
    df = add_tournament_metadata_columns(df)

    dummies = pd.get_dummies(df['tournament_type'], prefix='tt')
    df = pd.concat([df, dummies], axis=1)

    for t in df['tournament_type'].unique():
        mask = df['tournament_type'] == t
        col = f'{t}_mean'
        df[col] = df[mask].groupby('player_id')['points'].transform(lambda x: x.shift(1).expanding().mean())

    return df

def add_volatility_features(df):
    df = df.copy()
    df['rolling_std_10'] = df.groupby('player_id')['points'].transform(lambda x: x.shift(1).rolling(10).std())
    df['cv_10'] = df['rolling_std_10'] / (df['rolling_mean_10'] + 1e-6)
    return df

def add_peak_features(df):
    df = df.copy()

    df['rolling_max_10'] = df.groupby('player_id')['points'].transform(lambda x: x.shift(1).rolling(10).max())
    df['rolling_min_10'] = df.groupby('player_id')['points'].transform(lambda x: x.shift(1).rolling(10).min())
    return df

def add_tournament_skill_features(df):
    df = df.copy()
    for t in TOURNAMENT_TYPES:
        col_mean = f'{t}_mean'
        col_count = f'{t}_count'
        mask = df['tournament_type'] == t

        df[col_mean] = df[mask].groupby('player_id')['points'].transform(lambda x: x.shift(1).expanding().mean())
        df[col_count] = df[mask].groupby('player_id')['points'].transform(lambda x: x.shift(1).expanding().count())
    return df

def add_relative_features(df):
    df = df.copy()
    df['field_mean'] = df.groupby('date')['points'].transform('mean')
    df['field_std'] = df.groupby('date')['points'].transform('std')
    df['relative_score'] = (df['points'] - df['field_mean']) / (df['field_std'] + 1e-6)

    df['rolling_relative_5'] = df.groupby('player_id')['relative_score'].transform(lambda x: x.shift(1).rolling(5).mean())
    return df

def add_rank_features(df):
    df = df.copy()

    df['rank'] = df.groupby('date')['points'].rank(ascending=False)
    df['field_size'] = df.groupby('date')['player_id'].transform('count')
    if 'position' in df.columns:
        df['finish_position'] = df['position'].astype(float)
    else:
        df['finish_position'] = df['rank'].astype(float)
    df['normalized_finish'] = df['finish_position'] / df['field_size'].clip(lower=1)
    df['top_30_bubble_distance'] = df['finish_position'] - 30
    df['near_top_30_bubble'] = df['top_30_bubble_distance'].abs().le(5).astype(int)
    df['made_top_30'] = df['finish_position'].le(30).astype(int)
    df['bubble_points'] = np.where(
        df['top_30_bubble_distance'].between(-5, 5),
        df['points'],
        0,
    )
    df['rolling_rank_5'] = df.groupby('player_id')['rank'].transform(lambda x: x.shift(1).rolling(5).mean())
    df['rolling_finish_5'] = df.groupby('player_id')['finish_position'].transform(
        lambda x: x.shift(1).rolling(5).mean())
    df['rolling_top_30_rate_10'] = df.groupby('player_id')['made_top_30'].transform(
        lambda x: x.shift(1).rolling(10).mean())
    df['rolling_bubble_points_10'] = df.groupby('player_id')['bubble_points'].transform(
        lambda x: x.shift(1).rolling(10).mean())
    return df

def add_activity_features(df):
    df = df.copy()

    df['games_last_7_days'] = df.groupby('player_id')['date'].transform(lambda x: x.diff().dt.days.le(7).rolling(10).sum())
    return df

def add_streak_features(df):
    df = df.copy()
    df['positive'] = (df['points'] > 0).astype(int)
    def compute_streak(series):
        streak = []
        current = 0

        for val in series.shift(1).fillna(0):
            if val == 1:
                current += 1
            else:
                current = 0
            streak.append(current)
        return pd.Series(streak, index=series.index)
    df['win_streak'] = df.groupby('player_id')['positive'].transform(compute_streak)
    return df

def build_features(df):
    df = preprocess(df)
    df = add_basic_features(df)
    df = add_attendance_features(df)
    df = add_time_features(df)
    df = add_tournament_features(df)
    df = add_volatility_features(df)
    df = add_peak_features(df)
    df = add_relative_features(df)
    df = add_rank_features(df)
    df = add_activity_features(df)
    df = add_streak_features(df)
    df = add_tournament_skill_features(df)

    EXCLUDE_COLS = ['player_id', 'tournament_type', 'date']
    df = df.loc[:, ~df.columns.duplicated()]

    for col in df.columns:
        if col not in EXCLUDE_COLS:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)

    return df

def get_feature_columns(df):
    exclude = {
        'player_id', 'date', 'points', 'tournament_type',
        # Raw/result columns from the same event would leak the answer into the
        # training target. Keep their shifted/rolling derivatives instead.
        'position', 'bounties', 'old_points', 'placement_points', 'bounty_points',
        'gets_rating_points', 'rank', 'finish_position', 'normalized_finish',
        'top_30_bubble_distance', 'near_top_30_bubble', 'made_top_30',
        'bubble_points', 'field_mean', 'field_std', 'relative_score', 'positive',
    }
    return [col for col in df.columns if col not in exclude]

