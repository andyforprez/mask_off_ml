import pandas as pd

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

def save_averages(df, path='data/averages.txt'):
    df = df.copy()

    stats = df.groupby('player_id').agg(total_points=('points', 'sum'), games=('points', 'count'))
    stats['average'] = stats['total_points'] / stats['games']
    stats['average'] = stats['average'].fillna(0)
    stats = stats[stats['games'] >= 5]
    stats = stats.sort_values('average', ascending=False).reset_index()

    with open(path, 'w', encoding='utf-8') as f:
        for _, row in stats.iterrows():
            f.write(
                f'{row['player_id']} - avg: {row['average']:.2f} '
                f'(games: {row['games']})\n'
            )
    print('Averages Saved to data/averages.txt')
    return stats