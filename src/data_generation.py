import numpy as np
import pandas as pd


def generate_synthetic_load_data(n_customers=600, random_state=42):
    """Generate synthetic electricity consumption time-series with pattern clusters."""
    rng = np.random.default_rng(random_state)
    hours = np.arange(24)
    rows = []

    base_patterns = [
        # Residential daily profile: morning and evening peaks, low night load.
        np.array([0.5, 0.5, 0.4, 0.4, 0.5, 0.8, 1.2, 1.0, 0.9, 0.8, 0.7, 0.7,
                  0.8, 0.9, 0.9, 1.0, 1.1, 1.4, 1.6, 1.5, 1.2, 1.0, 0.8, 0.6]),
        # Commercial daily profile: work hours peak, low nights and weekends.
        np.array([0.4, 0.4, 0.5, 0.7, 1.0, 1.4, 1.6, 1.8, 1.8, 1.6, 1.4, 1.3,
                  1.2, 1.2, 1.2, 1.3, 1.4, 1.6, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5]),
        # Industrial profile: high constant baseline with a small afternoon bump.
        np.array([1.2, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.8, 1.8, 1.8, 1.8,
                  1.8, 1.9, 1.9, 1.9, 1.9, 1.9, 1.8, 1.8, 1.7, 1.6, 1.5, 1.4]),
        # Night-centric / low-demand cluster: more energy used late evening and night.
        np.array([0.7, 0.8, 0.9, 1.0, 1.0, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.9,
                  1.0, 1.1, 1.2, 1.3, 1.4, 1.4, 1.3, 1.3, 1.2, 1.1, 0.9, 0.8]),
    ]

    for i in range(n_customers):
        pattern_id = i % len(base_patterns)
        profile = base_patterns[pattern_id].copy()
        scale = rng.uniform(0.8, 1.4)
        noise = rng.normal(0.0, 0.08, size=hours.shape)
        variation = rng.normal(0.0, 0.05, size=hours.shape)
        daily_profile = np.clip(profile * scale + noise + variation, 0.1, None)

        row = {
            f'h{h}': daily_profile[h]
            for h in hours
        }
        row['household_id'] = f'H{1000 + i}'
        row['pattern_group'] = pattern_id
        rows.append(row)

    df = pd.DataFrame(rows)

    # Introduce missing values and noise to mimic real measurement faults.
    hour_cols = [f'h{h}' for h in hours]
    missing_mask = rng.random((len(df), len(hour_cols))) < 0.03
    for row_idx, col_idx in zip(*np.where(missing_mask)):
        df.iat[row_idx, col_idx] = np.nan

    df['avg_load'] = df[hour_cols].mean(axis=1)
    df['peak_load'] = df[[f'h{h}' for h in hours]].max(axis=1)
    df['morning_load'] = df[[f'h{h}' for h in range(6, 12)]].mean(axis=1)
    df['evening_load'] = df[[f'h{h}' for h in range(17, 23)]].mean(axis=1)
    df['load_variance'] = df[[f'h{h}' for h in hours]].var(axis=1)
    return df


def save_synthetic_data(path, n_customers=600, random_state=42):
    df = generate_synthetic_load_data(n_customers=n_customers, random_state=random_state)
    df.to_csv(path, index=False)
    return df


def load_consumption_data(path):
    return pd.read_csv(path)
