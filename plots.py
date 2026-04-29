import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.formatting.rule import ColorScaleRule, CellIsRule
from openpyxl.styles import PatternFill, Border, Side, Font, Alignment
from openpyxl.utils import get_column_letter


# ── 1. Cutoff vs Player ───────────────────────────────────────────────────────

def _compute_actual_cutoff_series(df):
    """Compute the 18th-place cumulative points at each historical date."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    all_dates  = sorted(df['date'].unique())

    last_pts = {}
    result   = []
    for d in all_dates:
        for _, row in df[df['date'] == d].iterrows():
            pid = row['player_id']
            last_pts[pid] = last_pts.get(pid, 0.0) + row['points']
        ranked = sorted(last_pts.values(), reverse=True)
        cutoff = ranked[17] if len(ranked) >= 18 else (ranked[-1] if ranked else 0)
        result.append({'date': d, 'cutoff': cutoff})
    return result


def plot_cutoff_vs_player(cutoff_history, expected_players, player_name,
                           player_path=None, df_actual=None):
    """
    Two-panel figure.
    Left:  full history (solid) + projected future (dashed) for both
           the tracked player and the 18th-place cutoff.
    Right: histogram of final cutoff values across all simulations,
           with the player's projected finish marked.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'Playoff Projection — {player_name}', fontsize=13, fontweight='bold')

    ax = axes[0]

    # ── Actual historical cutoff (solid blue) ──
    if df_actual is not None:
        actual_cutoff = _compute_actual_cutoff_series(df_actual)
        ac_dates  = pd.to_datetime([x['date'] for x in actual_cutoff])
        ac_vals   = [x['cutoff'] for x in actual_cutoff]
        ax.plot(ac_dates, ac_vals, color='steelblue', linewidth=2,
                label='18th place (actual)')

    # ── Projected cutoff: mean ± std band (dashed blue) ──
    if cutoff_history:
        rows = [
            {'date': pd.Timestamp(c['date']), 'cutoff': c['cutoff']}
            for sim in cutoff_history for c in sim
        ]
        cdf      = pd.DataFrame(rows)
        mean_cut = cdf.groupby('date')['cutoff'].mean()
        std_cut  = cdf.groupby('date')['cutoff'].std().fillna(0)

        # stitch from last actual date
        stitch_date = ac_dates[-1] if df_actual is not None else mean_cut.index[0]
        stitch_val  = ac_vals[-1]  if df_actual is not None else mean_cut.iloc[0]

        proj_dates = [stitch_date] + list(mean_cut.index)
        proj_vals  = [stitch_val]  + list(mean_cut.values)
        proj_std   = [0]           + list(std_cut.values)

        proj_dates = pd.to_datetime(proj_dates)
        proj_vals  = np.array(proj_vals)
        proj_std   = np.array(proj_std)

        ax.plot(proj_dates, proj_vals, color='steelblue', linewidth=2,
                linestyle='--', label='18th place (projected mean)')
        ax.fill_between(proj_dates, proj_vals - proj_std, proj_vals + proj_std,
                         color='steelblue', alpha=0.15, label='±1 std')

    # ── Actual player history (solid red) ──
    if df_actual is not None:
        df_actual2 = df_actual.copy()
        df_actual2['date'] = pd.to_datetime(df_actual2['date'])
        p_df = df_actual2[df_actual2['player_id'] == player_name].sort_values('date')
        if not p_df.empty:
            cum_pts = p_df['points'].cumsum()
            ax.plot(p_df['date'], cum_pts, color='crimson', linewidth=2,
                    label=f'{player_name} (actual)')
            last_actual_date = p_df['date'].iloc[-1]
            last_actual_pts  = float(cum_pts.iloc[-1])
        else:
            last_actual_date = None
            last_actual_pts  = None
    else:
        last_actual_date = None
        last_actual_pts  = None

    # ── Projected player path (dashed red) ──
    if player_path is not None and len(player_path) > 0:
        sim_dates = pd.to_datetime(list(player_path.index))
        sim_pts   = np.array(list(player_path.values))

        # stitch
        if last_actual_date is not None:
            stitch_d = [last_actual_date] + list(sim_dates)
            stitch_v = [last_actual_pts]  + list(sim_pts)
        else:
            stitch_d, stitch_v = list(sim_dates), list(sim_pts)

        ax.plot(pd.to_datetime(stitch_d), stitch_v, color='crimson', linewidth=2,
                linestyle='--', label=f'{player_name} (projected mean)')

    ax.axvline(pd.Timestamp(df_actual['date'].max()) if df_actual is not None else pd.Timestamp.now(),
               color='gray', linewidth=1, linestyle=':', alpha=0.7, label='Today')

    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Points')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.tick_params(axis='x', rotation=45)
    ax.set_title('Points Trajectory')

    # ── Panel 2: distribution of final cutoff ──
    ax2 = axes[1]
    if cutoff_history:
        final_cutoffs = np.array([sim[-1]['cutoff'] for sim in cutoff_history if sim])
        mu = float(np.mean(final_cutoffs))
        sd = float(np.std(final_cutoffs))

        ax2.hist(final_cutoffs, bins=40, color='steelblue', edgecolor='white',
                 linewidth=0.4, alpha=0.85, label='Cutoff distribution')
        ax2.axvline(mu, color='navy', linewidth=2, label=f'Mean: {mu:.0f}')
        ax2.axvline(mu - sd, color='navy', linewidth=1, linestyle='--', alpha=0.6)
        ax2.axvline(mu + sd, color='navy', linewidth=1, linestyle='--', alpha=0.6,
                    label=f'±1σ: {sd:.0f}')

        # Player projected finish
        if player_path is not None and len(player_path) > 0:
            proj_final = float(player_path.values[-1])
            if last_actual_pts is not None:
                proj_final += last_actual_pts   # add actual base
            ax2.axvline(proj_final, color='crimson', linewidth=2.5,
                        label=f'{player_name}: {proj_final:.0f}')
            pct = float(np.mean(final_cutoffs <= proj_final) * 100)
            ax2.set_title(f'Final Cutoff Distribution\n'
                          f'{player_name} clears cutoff in {pct:.1f}% of sims')
        else:
            ax2.set_title('Final Cutoff Distribution')

        ax2.set_xlabel('18th-place Points')
        ax2.set_ylabel('Frequency (simulations)')
        ax2.legend(fontsize=8)
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/cutoff_vs_player.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved data/cutoff_vs_player.png")


# ── 2. Rank Projections Multi ─────────────────────────────────────────────────

def plot_rank_projections_multi(real_ranks_dict, sim_ranks_dict, today, top_n=20):
    """
    Solid lines = actual rank history.
    Dashed lines = projected rank (mean across simulations).
    Playoff cutoff line at y=18.
    """
    players = list(real_ranks_dict.keys())[:top_n]

    fig, ax = plt.subplots(figsize=(14, 7))
    cmap   = plt.get_cmap('tab20')
    colors = {p: cmap(i % 20) for i, p in enumerate(players)}

    for player in players:
        color = colors[player]

        # Actual
        real = real_ranks_dict.get(player, [])
        r_dates = pd.to_datetime([x['date'] for x in real if x.get('rank') is not None])
        r_ranks = [x['rank'] for x in real if x.get('rank') is not None]

        if len(r_dates):
            ax.plot(r_dates, r_ranks, color=color, linewidth=1.6, alpha=0.9)

        # Projected
        sim = sim_ranks_dict.get(player)
        if sim is not None and len(sim) > 0:
            s_dates = pd.to_datetime(list(sim.index))
            s_ranks = list(sim.values)

            # stitch to last actual point
            if len(r_dates):
                s_dates = pd.to_datetime([r_dates[-1]] + list(s_dates))
                s_ranks = [r_ranks[-1]] + s_ranks

            ax.plot(s_dates, s_ranks, color=color, linewidth=1.6,
                    linestyle='--', alpha=0.9, label=player)

            # end label
            if len(s_dates):
                ax.annotate(player,
                            xy=(s_dates[-1], s_ranks[-1]),
                            xytext=(4, 0), textcoords='offset points',
                            fontsize=5.5, color=color, va='center', clip_on=True)

    ax.axvline(pd.Timestamp(today), color='black', linewidth=1.5,
               linestyle=':', alpha=0.6, label='Today')
    ax.axhline(18.5, color='limegreen', linewidth=1.5, linestyle='--', alpha=0.7,
               label='Playoff cutoff (18)')

    ax.invert_yaxis()
    ax.set_ylim(top_n + 2, 0)
    ax.set_xlabel('Date')
    ax.set_ylabel('Rank')
    ax.set_title(f'Rank Projections — Top {top_n} Players  (solid=actual, dashed=projected)')
    ax.legend(fontsize=5.5, ncol=3, loc='lower left')
    ax.grid(axis='y', alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('data/rank_projections.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved data/rank_projections.png")


# ── 3. Playoff Odds Excel ─────────────────────────────────────────────────────

def save_playoff_odds_excel(df, path='data/playoff_odds.xlsx'):
    """
    NHL-playoff-odds style workbook:
    Col A  — player name
    Col B  — Final Rank
    Col C  — Top 18 %  (green gradient)
    Col D+ — Rank 1 … Rank 18  (blue gradient)
    Red border below row 18.  Freeze D2.
    """
    df = df.copy()
    top_col = [c for c in df.columns if c.startswith('Top')][0]

    rank_cols_18 = sorted(
        [c for c in df.columns if c.startswith('Rank ') and int(c.split()[1]) <= 18],
        key=lambda c: int(c.split()[1])
    )
    df = df[[top_col] + rank_cols_18].copy()

    # expected rank (weighted)
    df['_exp'] = sum(df[f'Rank {i}'] * i for i in range(1, 19) if f'Rank {i}' in df.columns)
    df = df.sort_values(by=[top_col, '_exp'], ascending=[False, True]).drop(columns=['_exp'])
    df.insert(0, 'Final Rank', range(1, len(df) + 1))

    wb = Workbook()
    ws = wb.active
    ws.title = 'Playoff Odds'

    headers = ['Player', 'Final Rank', 'Top 18 %'] + [f'#{i}' for i in range(1, 19)]
    ws.append(headers)

    for player, row in df.iterrows():
        ws.append([player] + list(row.values))

    n_rows = len(df)
    n_cols = len(headers)

    # Header styling
    header_fill = PatternFill(start_color='1F3864', end_color='1F3864', fill_type='solid')
    header_font = Font(bold=True, color='FFFFFF', size=10)
    for col in range(1, n_cols + 1):
        cell = ws.cell(row=1, column=col)
        cell.fill   = header_fill
        cell.font   = header_font
        cell.alignment = Alignment(horizontal='center', vertical='center')

    # Format floats as percentages; zero → blank
    for row in ws.iter_rows(min_row=2, min_col=3, max_row=n_rows + 1, max_col=n_cols):
        for cell in row:
            if isinstance(cell.value, float):
                if cell.value == 0.0:
                    cell.value = None
                else:
                    cell.number_format = '0.0%'

    # Green gradient on Top-18 % column (C)
    ws.conditional_formatting.add(f'C2:C{n_rows + 1}', ColorScaleRule(
        start_type='num', start_value=0,   start_color='FF4444',
        mid_type='num',   mid_value=0.5,   mid_color='FFFF00',
        end_type='num',   end_value=1,     end_color='00CC44',
    ))

    # Blue gradient on per-rank columns (D+)
    last_col_letter = get_column_letter(n_cols)
    ws.conditional_formatting.add(f'D2:{last_col_letter}{n_rows + 1}', ColorScaleRule(
        start_type='min',        start_color='FFFFFF',
        mid_type='percentile',   mid_value=80, mid_color='9EC5FE',
        end_type='max',          end_color='0D6EFD',
    ))

    # Light green fill for Final Rank ≤ 18
    green_fill = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')
    ws.conditional_formatting.add(f'B2:B{n_rows + 1}', CellIsRule(
        operator='lessThanOrEqual', formula=['18'], fill=green_fill
    ))

    # Red border below playoff cutoff row (row 18 of data = spreadsheet row 19)
    red_side = Side(style='medium', color='FF0000')
    for col in range(1, n_cols + 1):
        ws.cell(row=19, column=col).border = Border(bottom=red_side)

    # Alternating row shading
    alt_fill = PatternFill(start_color='F5F5F5', end_color='F5F5F5', fill_type='solid')
    for r in range(2, n_rows + 2, 2):
        for c in range(1, n_cols + 1):
            ws.cell(row=r, column=c).fill = alt_fill

    # Column widths
    ws.column_dimensions['A'].width = 24
    ws.column_dimensions['B'].width = 10
    ws.column_dimensions['C'].width = 10
    for col_idx in range(4, n_cols + 1):
        ws.column_dimensions[get_column_letter(col_idx)].width = 7

    ws.row_dimensions[1].height = 22
    ws.freeze_panes = 'D2'

    wb.save(path)
    print(f"Saved {path}")