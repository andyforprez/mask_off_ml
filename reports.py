import pandas as pd
import numpy as np


def save_rankings(df, target_date, path='data/rankings.txt'):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    target_date = pd.to_datetime(target_date)
    df = df[df['date'] <= target_date]

    standings = df.groupby('player_id').agg(points=('points', 'sum'), games=('points', 'count'))
    standings = standings.sort_values('points', ascending=False)
    standings['rank'] = standings['points'].rank(ascending=False, method='min')
    standings = standings.reset_index()

    with open(path, 'w', encoding='utf-8') as f:
        for _, row in standings.iterrows():
            f.write(f'{int(row['rank'])}. {row['player_id']} - {row['points']} pts ({int(row['games'])} games)\n')
    print('Rankings Saved to data/rankings.txt')
    return standings


def save_averages(df, path='data/averages.txt', min_games=2):
    df = df.copy()

    stats = df.groupby('player_id').agg(total_points=('points', 'sum'), games=('points', 'count'))
    stats['average'] = stats['total_points'] / stats['games']
    stats['average'] = stats['average'].fillna(0)
    stats = stats[stats['games'] >= min_games]
    stats = stats.sort_values('average', ascending=False).reset_index()

    with open(path, 'w', encoding='utf-8') as f:
        for _, row in stats.iterrows():
            f.write(
                f'{row['player_id']} - avg: {row['average']:.2f} '
                f'(games: {row['games']})\n'
            )
    print('Averages Saved to data/averages.txt')
    return stats


def build_player_profiles(df, bubble_rank=30, bubble_window=5):
    """Build player-level performance profiles, including top-30 bubble stats."""

    df = df.copy()
    if 'position' not in df.columns:
        df['position'] = df.groupby('date')['points'].rank(ascending=False, method='first')

    df['position'] = pd.to_numeric(df['position'], errors='coerce')
    df['points'] = pd.to_numeric(df['points'], errors='coerce').fillna(0)
    df['made_top_9'] = df['position'] <= 9
    df['made_top_20'] = df['position'] <= 20
    df['made_top_30'] = df['position'] <= 30
    df['near_top_30_bubble'] = df['position'].between(bubble_rank - bubble_window, bubble_rank + bubble_window)
    df['cashed_top_30_bubble'] = df['near_top_30_bubble'] & df['made_top_30']

    grouped = df.groupby('player_id')
    profiles = grouped.agg(
        games=('points', 'count'),
        total_points=('points', 'sum'),
        avg_points=('points', 'mean'),
        avg_finish=('position', 'mean'),
        top_9_rate=('made_top_9', 'mean'),
        top_20_rate=('made_top_20', 'mean'),
        top_30_rate=('made_top_30', 'mean'),
        made_top_30_games=('made_top_30', 'sum'),
        near_top_30_bubble_games=('near_top_30_bubble', 'sum'),
    )

    bubble_rows = df[df['near_top_30_bubble']]
    if bubble_rows.empty:
        profiles['top_30_bubble_avg_points'] = 0.0
        profiles['top_30_bubble_conversion_rate'] = 0.0
    else:
        bubble_grouped = bubble_rows.groupby('player_id')
        profiles['top_30_bubble_avg_points'] = bubble_grouped['points'].mean()
        profiles['top_30_bubble_conversion_rate'] = bubble_grouped['made_top_30'].mean()

    profiles['points_per_top_30'] = np.divide(
        profiles['total_points'].astype(float),
        profiles['made_top_30_games'].astype(float),
        out=np.zeros(len(profiles), dtype=float),
        where=profiles['made_top_30_games'].to_numpy(dtype=float) != 0,
    )
    profiles['top_30_bubble_avg_points'] = profiles['top_30_bubble_avg_points'].fillna(0)
    profiles['top_30_bubble_conversion_rate'] = profiles['top_30_bubble_conversion_rate'].fillna(0)
    profiles['points_per_top_30'] = profiles['points_per_top_30'].fillna(0)
    profiles['made_top_30_games'] = profiles['made_top_30_games'].astype(int)
    profiles['near_top_30_bubble_games'] = profiles['near_top_30_bubble_games'].astype(int)

    # A simple descriptive score for profile sorting: rewards consistent top-30
    # finishes and actual conversion when the player lands near the cutoff.
    profiles['bubble_score'] = (
        0.6 * profiles['top_30_rate']
        + 0.4 * profiles['top_30_bubble_conversion_rate']
    )

    profiles = profiles.reset_index().sort_values(
        ['bubble_score', 'top_30_rate', 'avg_points'], ascending=False
    )
    return profiles


def save_player_profiles(df, path='data/player_profiles.csv'):
    profiles = build_player_profiles(df)
    profiles.to_csv(path, index=False)
    print(f'Player profiles saved to {path}')
    return profiles