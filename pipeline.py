import pandas as pd
import numpy as np
import warnings
from features import build_features, get_feature_columns


# ── Feature snapshot (computed once) ─────────────────────────────────────────

def build_player_snapshots(df, model, feature_cols):
    """
    Compute each player's feature vector from their most recent game row.
    Returns dict: player_id -> np.array of shape (n_features,)
    """
    df_features = build_features(df)
    latest = (
        df_features.sort_values('date')
        .groupby('player_id')
        .tail(1)
        .copy()
    )
    for col in feature_cols:
        if col not in latest.columns:
            latest[col] = 0.0
    snapshots = {}
    for _, row in latest.iterrows():
        pid = row['player_id']
        vec = row[feature_cols].values.astype(float)
        snapshots[pid] = vec

    return snapshots


def predict_from_snapshots(snapshots, model, noise_std=0.0):
    """
    Batch-predict points for all players from cached feature arrays.
    Returns dict: player_id -> predicted_points (float >= 0)
    """
    players = list(snapshots.keys())
    X = np.array([snapshots[p] for p in players])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = model.predict(X)

    if noise_std > 0:
        preds = preds + np.random.normal(0, noise_std, size=len(preds))

    preds = np.maximum(preds, 0)
    return dict(zip(players, preds.tolist()))


# ── Single simulation run ─────────────────────────────────────────────────────

def simulate_one_run(start_standings, snapshots, model,
                     schedule, noise_std, inactive_players=None, cutoff_rank=18):
    """
    Simulate one Monte Carlo path.

    start_standings: dict player -> current cumulative points (real historical total)
    Returns: (cutoff_series, player_series)
        cutoff_series: list of {'date', 'cutoff'}
        player_series: dict player -> list of {'date', 'points', 'rank'}
    """
    standings = start_standings.copy()
    inactive  = set(inactive_players or [])

    cutoff_series = []
    player_series = {}

    for date, t_type in schedule:
        # Get predicted *incremental* points for this single game day
        day_preds = predict_from_snapshots(snapshots, model, noise_std)

        # Add predicted points to running standings (skip inactive)
        for player, pts in day_preds.items():
            if player in inactive:
                continue
            standings[player] = standings.get(player, 0.0) + pts

        # Rank everyone
        ranked = sorted(standings.items(), key=lambda x: x[1], reverse=True)
        cutoff_idx = cutoff_rank - 1
        cutoff_val = ranked[cutoff_idx][1] if len(ranked) >= cutoff_rank else 0.0

        cutoff_series.append({'date': date, 'cutoff': cutoff_val})

        rank_lookup = {p: r + 1 for r, (p, _) in enumerate(ranked)}
        for player, total_pts in standings.items():
            if player not in player_series:
                player_series[player] = []
            player_series[player].append({
                'date':   date,
                'points': total_pts,
                'rank':   rank_lookup.get(player, len(standings)),
            })

    return cutoff_series, player_series


# ── Main simulation runner ────────────────────────────────────────────────────

def run_simulations(df, model, schedule, n_sim=500, noise_std=150,
                    inactive_players=None, cutoff_rank=18):
    """
    Pre-computes feature snapshots once, then runs n_sim Monte Carlo paths fast.
    """
    feature_cols = list(getattr(model, 'feature_names_', None) or
                        get_feature_columns(build_features(df)))

    print("Building player snapshots (one-time)...")
    snapshots = build_player_snapshots(df, model, feature_cols)
    print(f"  Snapshots ready for {len(snapshots)} players.")

    # Current cumulative standings from actual data
    start_standings = (
        df.groupby('player_id')['points']
        .sum()
        .to_dict()
    )

    cutoff_history = []
    player_history = []
    log_every      = max(1, n_sim // 10)

    for i in range(n_sim):
        if (i + 1) % log_every == 0:
            print(f"  Simulation {i + 1}/{n_sim}...")

        c_series, p_series = simulate_one_run(
            start_standings, snapshots, model,
            schedule, noise_std, inactive_players,
            cutoff_rank=cutoff_rank
        )
        cutoff_history.append(c_series)
        player_history.append(p_series)

    return cutoff_history, player_history


# ── Aggregation helpers ───────────────────────────────────────────────────────

def compute_expected_cutoff(cutoff_history):
    """Returns dict: date -> mean cutoff across all simulations."""
    rows = [
        {'date': c['date'], 'cutoff': c['cutoff']}
        for sim in cutoff_history
        for c in sim
    ]
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    return df.groupby('date')['cutoff'].mean().to_dict()


def compute_expected_player_points(player_history):
    """Returns Series: player_id -> mean final cumulative points, sorted desc."""
    final_pts = {}
    for sim in player_history:
        for player, series in sim.items():
            if series:
                final_pts.setdefault(player, []).append(series[-1]['points'])
    avg = {p: float(np.mean(v)) for p, v in final_pts.items()}
    return pd.Series(avg).sort_values(ascending=False)


def compute_expected_player_path(player_history, player_id):
    """Returns Series: date -> mean cumulative points for one player."""
    rows = []
    for sim in player_history:
        if player_id in sim:
            for entry in sim[player_id]:
                rows.append({'date': entry['date'], 'points': entry['points']})
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    return df.groupby('date')['points'].mean()


def compute_expected_player_rank(player_history, player_id):
    """Returns Series: date -> mean rank for one player."""
    rows = []
    for sim in player_history:
        if player_id in sim:
            for entry in sim[player_id]:
                rows.append({'date': entry['date'], 'rank': entry['rank']})
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    return df.groupby('date')['rank'].mean()

def compute_sample_multiplier(games_played, min_games_for_full=10, min_multiplier=0.22):
    if games_played is None or games_played <= 0:
        return float(min_multiplier)
    if games_played >= min_games_for_full:
        return 1.0
    min_multiplier = max(0.0, min(1.0, float(min_multiplier)))
    span = 1.0 - min_multiplier
    return float(min_multiplier + (games_played / min_games_for_full) * span)

def compute_season_progress(games_played_dict, min_games_for_full=10):
    if not games_played_dict:
        return 0.0
    counts = list(games_played_dict.values())
    median_games = float(np.median(counts))
    return min(1.0, median_games / min_games_for_full)

def compute_playoff_odds(player_history, cutoff=18, eval_pool=50, games_played=None, min_games_for_full=10, min_multiplier=0.22):
    """
    Returns DataFrame with rows = top eval_pool players (by mean final pts),
    columns = Rank 1..cutoff + 'Top N Prob'.
    """
    # Determine top eval_pool players
    final_pts = {}
    for sim in player_history:
        for player, series in sim.items():
            if series:
                final_pts.setdefault(player, []).append(series[-1]['points'])

    avg_pts     = {p: float(np.mean(v)) for p, v in final_pts.items()}
    all_players = sorted(avg_pts, key=avg_pts.get, reverse=True)
    all_set     = set(all_players)

    rank_counts = {p: [0] * eval_pool for p in all_players}
    n_sim       = len(player_history)

    for sim in player_history:
        final = {p: sim[p][-1]['points'] for p in sim if sim[p]}
        ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)
        for rank, (player, _) in enumerate(ranked):
            if player in all_set and rank < eval_pool:
                rank_counts[player][rank] += 1

    df = pd.DataFrame(rank_counts, index=[f'Rank {i+1}' for i in range(eval_pool)]).T
    df = df / n_sim
    df = df.iloc[:, :cutoff]
    raw_top_prob = df.sum(axis=1)

    if games_played is not None:
        multipliers = pd.Series({
            player: compute_sample_multiplier(games_played.get(player, 0), min_games_for_full, min_multiplier)
            for player in df.index
        })
    else:
        multipliers = pd.Series(1.0, index=df.index)

    sim_adjusted = (raw_top_prob * multipliers).clip(lower=0, upper=1)

    season_progress = compute_season_progress(games_played, min_games_for_full) if games_played else 0.0
    season_newness = (1.0 - season_progress) ** 2

    total_players = len(all_players) if all_players else 1
    rank_weights = pd.Series({
        p: ((total_players - i) / total_players) ** 1.5 for i, p in enumerate(all_players)
    }, dtype=float)
    floor_max = (cutoff / max(total_players, cutoff)) * 0.8
    prob_floor = rank_weights * floor_max * season_newness
    blended = sim_adjusted.combine(prob_floor, max).clip(upper=1)

    display_threshold = 0.0005
    rank_cols = df.columns.tolist()
    for player in df.index:
        raw = sim_adjusted[player]
        target = blended[player]
        if target < display_threshold:
            df.loc[player, rank_cols] = 0.0
        elif raw > 0:
            df.loc[player, rank_cols] *= (target / raw)
        else:
            df.loc[player, rank_cols] = 0.0
            df.loc[player, rank_cols[-1]] = target


    col = f'Top {cutoff} Prob'
    df[col] = blended.where(blended >= display_threshold, 0.0)
    df = df.sort_values(col, ascending=False)
    df.index.name = 'player_id'
    return df


def compute_expected_final_ranking(player_history, eval_pool=50, output_top=50):
    """Returns list of (player, avg_rank) sorted by avg_rank ascending."""
    rank_data = {}
    for sim in player_history:
        final = {p: sim[p][-1]['points'] for p in sim if sim[p]}
        ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)
        for i, (player, _) in enumerate(ranked[:eval_pool]):
            rank_data.setdefault(player, []).append(i + 1)

    avg_rank = {p: float(np.mean(v)) for p, v in rank_data.items()}
    return sorted(avg_rank.items(), key=lambda x: x[1])[:output_top]