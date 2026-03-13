import numpy as np
import pandas as pd


def generate_promo_indicators(dates, promo_config, seed=None):
    """Generate 0/1 promo indicator column and its revenue contribution.

    Places promo occurrences at random dates within the date range,
    each lasting `duration_days`. For weekly frequency, a week gets 1
    if any day in that week had a promo.

    Args:
        dates: DatetimeIndex of the scenario.
        promo_config: PromoConfig with name, coefficient, n_occurrences, duration_days.
        seed: Random seed.

    Returns:
        Tuple of (indicator_array, contribution_array).
        indicator_array: 0/1 array of length len(dates).
        contribution_array: coefficient * indicator_array.
    """
    rng = np.random.default_rng(seed)
    n = len(dates)
    indicator = np.zeros(n)

    # Determine total span in days
    start = dates[0]
    end = dates[-1]
    total_days = (end - start).days

    if total_days <= 0:
        return indicator, indicator * promo_config.coefficient

    # Determine the frequency (daily or weekly)
    freq_days = 1
    if n > 1:
        median_gap = np.median(np.diff(dates).astype('timedelta64[D]').astype(int))
        freq_days = max(int(median_gap), 1)

    # Pick random start days for each occurrence
    n_years = max(1, total_days / 365)
    total_occurrences = max(1, int(promo_config.n_occurrences * n_years))

    for _ in range(total_occurrences):
        # Random start day within range
        promo_start_offset = rng.integers(0, max(1, total_days - promo_config.duration_days))
        promo_start = start + pd.Timedelta(days=int(promo_start_offset))
        promo_end = promo_start + pd.Timedelta(days=promo_config.duration_days - 1)

        # Mark periods that overlap with this promo window
        for j in range(n):
            period_start = dates[j]
            period_end = period_start + pd.Timedelta(days=freq_days - 1)
            if period_start <= promo_end and period_end >= promo_start:
                indicator[j] = 1

    contribution = promo_config.coefficient * indicator
    return indicator, contribution


def add_promos_legacy(df, promos, noise_level=0.2):
    """Add promotional effects to revenue data (legacy engine).

    Applies lift percentages to revenue columns during promo date ranges,
    with noise. Adds indicator columns and incremental revenue columns.

    Args:
        df: DataFrame with 'date' column and revenue columns.
        promos: List of promo configs, each with 'name' and 'dates' list.
            Each date entry has 'start_date', 'end_date', 'lift_percentage'.
        noise_level: Random variation (0-1).

    Returns:
        DataFrame with promo effects applied.
    """
    df = df.copy()
    revenue_cols = [col for col in df.columns if col.endswith('_revenue')]
    df['promo'] = 0

    for promo in promos:
        name = promo['name']
        df[name] = 0
        df[f'{name}_revenue'] = 0.0

        for period in promo['dates']:
            start = pd.to_datetime(period['start_date'])
            end = pd.to_datetime(period['end_date'])
            lift = period['lift_percentage']

            mask = (df['date'] >= start) & (df['date'] <= end)
            df.loc[mask, name] = 1
            df.loc[mask, 'promo'] = 1

            for col in revenue_cols:
                baseline = df.loc[mask, col].values
                noise = np.random.uniform(1 - noise_level, 1 + noise_level, size=len(baseline))
                promo_rev = baseline * (1 + lift) * noise
                incremental = promo_rev - baseline
                df.loc[mask, f'{name}_revenue'] = df.loc[mask, f'{name}_revenue'].values + incremental
                df.loc[mask, col] = promo_rev

    return df
