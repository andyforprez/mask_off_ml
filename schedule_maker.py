import pandas as pd

TOURNAMENT_SCHEDULE = {
    2: 'double rating points',
    3: 'high roller',
    4: 'deep classic',
    5: 'bounty',
    6: 'triple shot'
}

def build_calendar(start_date, end_date):
    dates = pd.date_range(start_date, end_date)

    schedule = []

    for d in dates:
        if d.weekday() in TOURNAMENT_SCHEDULE:
            schedule.append((d, TOURNAMENT_SCHEDULE[d.weekday()]))
    return schedule