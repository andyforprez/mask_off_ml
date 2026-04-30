import pandas as pd

from features import build_features, get_feature_columns
from model import train_model
from reports import save_rankings, save_averages
from schedule_maker import build_calendar
from pipeline import (
    run_simulations,
    compute_expected_cutoff,
    compute_expected_player_points,
    compute_expected_player_path,
    compute_expected_player_rank,
    compute_playoff_odds,
    compute_expected_final_ranking,
)
from plots import (
    plot_cutoff_vs_player,
    plot_rank_projections_multi,
    save_playoff_odds_excel,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
PLAYER_TO_TRACK  = 'Антуан Гризманн'
INACTIVE_PLAYERS = ['Дядя Витя', 'Кесадилия', 'Комарик', 'E_ishutkin']
N_SIM            = 1000
NOISE_STD        = 200
EVAL_POOL        = 100
TOP_N_RANK_PLOT  = 18
# ─────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_actual_rank_path(df, player_id):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    all_dates = sorted(df['date'].unique())

    last_pts = {}
    result = []
    for d in all_dates:
        for _, row in df[df['date'] == d].iterrows():
            pid = row['player_id']
            last_pts[pid] = last_pts.get(pid, 0) + row['points']
        ranked = sorted(last_pts.items(), key=lambda x: x[1], reverse=True)
        rank_lookup = {p: r + 1 for r, (p, _) in enumerate(ranked)}
        result.append({'date': d, 'rank': rank_lookup.get(player_id)})
    return result


def save_expected_ranking(final_ranking, path='data/expected_final_ranking.txt'):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('EXPECTED FINAL RANKING (FROM SIMULATION)\n')
        f.write('=' * 50 + '\n\n')
        for i, (player, avg_rank) in enumerate(final_ranking, 1):
            f.write(f'{i}. {player} - avg rank: {avg_rank:.2f}\n')
    print(f"Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

df = pd.read_csv('data/raw_data.csv')
df['date'] = pd.to_datetime(df['date'])
today = df['date'].max()

# 1. Leaderboard files
save_rankings(df, today)
save_averages(df)

# 2. Train model
print("\n── Training model ──")
df_features = build_features(df)
feature_cols = get_feature_columns(df_features)
model = train_model(df_features, feature_cols)

# 3. Future schedule
next_day = str((today + pd.Timedelta(days=1)).date())
future_schedule = build_calendar(
    start_date=next_day,
    end_date='2026-05-01',
    overrides={pd.Timestamp('2026-05-01'): 'bounty'}
)

if not future_schedule:
    print("No future games — nothing to simulate.")
else:
    print(f"\nFuture schedule: {len(future_schedule)} game days")
    for d, t in future_schedule:
        print(f"  {d}  {t}")

    # 4. Simulations
    print(f"\n── Running {N_SIM} simulations ──")
    cutoff_history, player_history = run_simulations(
        df, model, future_schedule,
        n_sim=N_SIM,
        noise_std=NOISE_STD,
        inactive_players=INACTIVE_PLAYERS,
    )

    # 5. Summaries
    expected_players = compute_expected_player_points(player_history)
    expected_cutoff  = compute_expected_cutoff(cutoff_history)
    final_cutoff_val = list(expected_cutoff.values())[-1] if expected_cutoff else 0

    print("\n── Expected final top-20 ──")
    print(expected_players.head(20).to_string())
    print(f"\nExpected cutoff (18th place): {final_cutoff_val:.0f} pts")

    # 6. Cutoff vs player plot
    player_sim_path = compute_expected_player_path(player_history, PLAYER_TO_TRACK)
    plot_cutoff_vs_player(
        cutoff_history, expected_players, PLAYER_TO_TRACK,
        player_path=player_sim_path,
        df_actual=df,
    )

    # 7. Rank projections
    current_standings = df.groupby('player_id')['points'].sum().sort_values(ascending=False)
    top_players = list(current_standings.index[:TOP_N_RANK_PLOT])

    print("\nBuilding actual rank paths...")
    actual_rank_paths = {p: get_actual_rank_path(df, p) for p in top_players}
    sim_rank_paths    = {p: compute_expected_player_rank(player_history, p) for p in top_players}

    plot_rank_projections_multi(actual_rank_paths, sim_rank_paths, today, top_n=TOP_N_RANK_PLOT)

    # 8. expected_final_ranking.txt
    final_ranking = compute_expected_final_ranking(player_history, eval_pool=EVAL_POOL, output_top=EVAL_POOL)
    save_expected_ranking(final_ranking)

    # 9. playoff_odds.csv + .xlsx
    odds_df = compute_playoff_odds(player_history, cutoff=18, eval_pool=EVAL_POOL)
    odds_df.to_csv('data/playoff_odds.csv')
    print("Saved data/playoff_odds.csv")
    save_playoff_odds_excel(odds_df, 'data/playoff_odds.xlsx')