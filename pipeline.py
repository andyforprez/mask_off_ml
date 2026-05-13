import pandas as pd
import numpy as np
import warnings
from conversion import build_bounty_expectations, sample_bounty_bonus, score_ranked_players
from features import build_features, get_feature_columns


# ── Attendance model ──────────────────────────────────────────────────────────

def build_attendance_model(df):
    """
    Per-player, per-tournament-type attendance probability.

    Logic: smoothed observed rate = appearances / total_runs_of_that_type,
    blended with the player's own overall attendance rate as a Bayesian prior.
    Hard floor of 20% for anyone who showed up at least once (they're clearly
    an active member); cap at 95% to keep some variance.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # How many distinct days each tournament type ran
    tourney_runs = (
        df[['date', 'tournament_type']]
        .drop_duplicates()
        .groupby('tournament_type')['date']
        .count()
        .to_dict()
    )

    # Per player per tourney: how many days did they show up
    player_tourney = (
        df.groupby(['player_id', 'tournament_type'])['date']
        .nunique()
        .reset_index(name='appearances')
    )

    # Total distinct game days each player appeared
    player_total_days = df.groupby('player_id')['date'].nunique().to_dict()
    total_game_days   = df['date'].nunique()

    attendance = {}
    overall    = {}

    for player in df['player_id'].unique():
        player_days = player_total_days.get(player, 0)

        # Smoothed overall rate: weak prior of 2 pseudo-games at 0.5
        prior_weight   = 2
        overall[player] = (player_days + prior_weight * 0.5) / (total_game_days + prior_weight)

        attendance[player] = {}
        player_rows = player_tourney[player_tourney['player_id'] == player]

        for _, row in player_rows.iterrows():
            t           = row['tournament_type']
            appearances = row['appearances']
            total_runs  = tourney_runs.get(t, 1)
            p_overall   = overall[player]

            # Bayesian smoothing: pull toward player's own overall rate
            # prior_n = 2 pseudo-observations at their overall rate
            prior_n  = 2
            smoothed = (appearances + prior_n * p_overall) / (total_runs + prior_n)

            # Floor 20% for known players, cap at 95%
            attendance[player][t] = float(np.clip(smoothed, 0.20, 0.95))

    return attendance, overall


def get_attendance_prob(player, tournament_type, attendance_model, overall_model):
    """
    Probability that `player` attends a given tournament type in the future.
    """
    att_by_type = attendance_model.get(player, {})

    if tournament_type in att_by_type:
        return float(att_by_type[tournament_type])

    # Played before but never this tourney type: 75% of their overall rate
    overall = overall_model.get(player, 0.15)
    return float(np.clip(overall * 0.75, 0.10, 0.80))


# ── Feature snapshot ──────────────────────────────────────────────────────────

def build_player_snapshots(df, model, feature_cols):
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
        snapshots[row['player_id']] = row[feature_cols].values.astype(float)
    return snapshots


def predict_from_snapshots(snapshots, model, attending_players, noise_std=0.0):
    """Predict incremental points only for players who are attending."""
    players = [p for p in attending_players if p in snapshots]
    if not players:
        return {}

    X = np.array([snapshots[p] for p in players])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = model.predict(X)

    if noise_std > 0:
        preds = preds + np.random.normal(0, noise_std, size=len(preds))

    return dict(zip(players, np.maximum(preds, 0).tolist()))


# ── Single simulation run ─────────────────────────────────────────────────────

def simulate_one_run(start_standings, snapshots, model,
                     schedule, noise_std, attendance_model, overall_model,
                     inactive_players=None, cutoff_rank=24):
    standings    = start_standings.copy()
    inactive     = set(inactive_players or [])
    all_players  = list(snapshots.keys())
    cutoff_series = []
    player_series = {}

    for date, t_type in schedule:
        attending = [
            p for p in all_players
            if p not in inactive
            and np.random.random() < get_attendance_prob(
                p, t_type, attendance_model, overall_model
            )
        ]

        day_preds = predict_from_snapshots(snapshots, model, attending, noise_std)
        for player, pts in day_preds.items():
            standings[player] = standings.get(player, 0.0) + pts

        ranked    = sorted(standings.items(), key=lambda x: x[1], reverse=True)
        cutoff_val = ranked[cutoff_rank - 1][1] if len(ranked) >= cutoff_rank else 0.0
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
                    inactive_players=None, cutoff_rank=24):
    feature_cols = list(getattr(model, 'feature_names_', None) or
                        get_feature_columns(build_features(df)))

    print("Building player snapshots (one-time)...")
    snapshots = build_player_snapshots(df, model, feature_cols)
    print(f"  Snapshots ready for {len(snapshots)} players.")

    print("Building attendance model (one-time)...")
    attendance_model, overall_model = build_attendance_model(df)
    print(f"  Attendance model ready for {len(attendance_model)} players.")

    start_standings = df.groupby('player_id')['points'].sum().to_dict()
    cutoff_history  = []
    player_history  = []
    log_every       = max(1, n_sim // 10)

    for i in range(n_sim):
        if (i + 1) % log_every == 0:
            print(f"  Simulation {i + 1}/{n_sim}...")
        c_series, p_series = simulate_one_run(
            start_standings, snapshots, model, schedule, noise_std,
            attendance_model, overall_model,
            inactive_players=inactive_players, cutoff_rank=cutoff_rank,
        )
        cutoff_history.append(c_series)
        player_history.append(p_series)

    return cutoff_history, player_history


# ── Aggregation helpers ───────────────────────────────────────────────────────

def compute_expected_cutoff(cutoff_history):
    rows = [{'date': c['date'], 'cutoff': c['cutoff']}
            for sim in cutoff_history for c in sim]
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    return df.groupby('date')['cutoff'].mean().to_dict()


def compute_expected_player_points(player_history):
    final_pts = {}
    for sim in player_history:
        for player, series in sim.items():
            if series:
                final_pts.setdefault(player, []).append(series[-1]['points'])
    avg = {p: float(np.mean(v)) for p, v in final_pts.items()}
    return pd.Series(avg).sort_values(ascending=False)


def compute_expected_player_path(player_history, player_id):
    rows = []
    for sim in player_history:
        if player_id in sim:
            for entry in sim[player_id]:
                rows.append({'date': entry['date'], 'points': entry['points']})
    if not rows:
        return pd.Series(dtype=float)
    return pd.DataFrame(rows).groupby('date')['points'].mean()


def compute_expected_player_rank(player_history, player_id):
    rows = []
    for sim in player_history:
        if player_id in sim:
            for entry in sim[player_id]:
                rows.append({'date': entry['date'], 'rank': entry['rank']})
    if not rows:
        return pd.Series(dtype=float)
    return pd.DataFrame(rows).groupby('date')['rank'].mean()


def compute_sample_multiplier(games_played, min_games_for_full=10, min_multiplier=0.22):
    if games_played is None or games_played <= 0:
        return float(min_multiplier)
    if games_played >= min_games_for_full:
        return 1.0
    min_multiplier = max(0.0, min(1.0, float(min_multiplier)))
    return float(min_multiplier + (games_played / min_games_for_full) * (1.0 - min_multiplier))


def compute_season_progress(games_played_dict, min_games_for_full=10):
    if not games_played_dict:
        return 0.0
    return min(1.0, float(np.median(list(games_played_dict.values()))) / min_games_for_full)


def compute_playoff_odds(player_history, cutoff=24, eval_pool=50, games_played=None,
                         min_games_for_full=10, min_multiplier=0.22):
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
        final  = {p: sim[p][-1]['points'] for p in sim if sim[p]}
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
            p: compute_sample_multiplier(games_played.get(p, 0), min_games_for_full, min_multiplier)
            for p in df.index
        })
    else:
        multipliers = pd.Series(1.0, index=df.index)

    sim_adjusted    = (raw_top_prob * multipliers).clip(0, 1)
    season_progress = compute_season_progress(games_played, min_games_for_full) if games_played else 0.0
    season_newness  = (1.0 - season_progress) ** 2
    total_players   = len(all_players) or 1

    rank_weights = pd.Series({
        p: ((total_players - i) / total_players) ** 1.5
        for i, p in enumerate(all_players)
    }, dtype=float)
    floor_max  = (cutoff / max(total_players, cutoff)) * 0.8
    prob_floor = rank_weights * floor_max * season_newness
    blended    = sim_adjusted.combine(prob_floor, max).clip(upper=1)

    display_threshold = 0.0005
    rank_cols = df.columns.tolist()
    for player in df.index:
        raw    = sim_adjusted[player]
        target = blended[player]
        if target < display_threshold:
            df.loc[player, rank_cols] = 0.0
        elif raw > 0:
            df.loc[player, rank_cols] *= (target / raw)
        else:
            df.loc[player, rank_cols] = 0.0
            df.loc[player, rank_cols[-1]] = target

    df[rank_cols] = df[rank_cols].clip(lower=0.0)
    col = f'Top {cutoff} Prob'
    df[col] = blended.where(blended >= display_threshold, 0.0)
    for player in df.index:
        target  = df.at[player, col]
        row_sum = float(df.loc[player, rank_cols].sum())
        if target < display_threshold:
            df.loc[player, rank_cols] = 0.0
        elif row_sum > 0:
            df.loc[player, rank_cols] *= (target / row_sum)
        else:
            df.loc[player, rank_cols] = 0.0
            df.loc[player, rank_cols[-1]] = target

    df = df.sort_values(col, ascending=False)
    df.index.name = 'player_id'
    return df


def compute_expected_final_ranking(player_history, eval_pool=50, output_top=50):
    rank_data = {}
    for sim in player_history:
        final  = {p: sim[p][-1]['points'] for p in sim if sim[p]}
        ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)
        for i, (player, _) in enumerate(ranked[:eval_pool]):
            rank_data.setdefault(player, []).append(i + 1)
    avg_rank = {p: float(np.mean(v)) for p, v in rank_data.items()}
    return sorted(avg_rank.items(), key=lambda x: x[1])[:output_top]