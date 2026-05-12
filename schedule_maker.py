import pandas as pd

DEFAULT_SCHEDULE = {
    1: 'high roller',
    2: 'double rating points',
    3: 'phoenix',
    4: 'deep classic',
    5: 'bounty',
    6: 'triple shot'
}

def build_calendar(start_date, end_date, overrides=None):
    dates = pd.date_range(start_date, end_date)

    schedule = []

    normalized_overrides = {}
    if overrides:
        normalized_overrides = {pd.Timestamp(k): v for k, v in overrides.items()}

    for d in dates:
        if d.weekday() in DEFAULT_SCHEDULE:
            t_type = DEFAULT_SCHEDULE[d.weekday()]
            if d in normalized_overrides:
                t_type = normalized_overrides[d]
            schedule.append((d, t_type))
    return schedule