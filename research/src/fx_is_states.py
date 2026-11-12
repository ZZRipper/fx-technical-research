"""Fixed descriptive entry-state bins and causal BASE-performance inputs.

No threshold in this module is fitted to outcomes. State summaries are labels
for research; a filter is a separate, subsequently frozen decision.
"""
import numpy as np
import pandas as pd


STATE_BINS = {
    'vol_ratio_20_63': [0.8, 1.0, 1.25],
    'realized_vol_20_pct': [5.0, 10.0, 15.0],
    'efficiency_63': [0.1, 0.2, 0.4],
    'direction_return_20_vol': [-1.0, 0.0, 1.0],
    'direction_distance_ma_60_vol': [-1.0, 0.0, 1.0],
    'K_vix': [15.0, 20.0, 30.0],
    'K_vix_d5': [-3.0, 0.0, 3.0],
    'direction_spread_bp': [-100.0, 0.0, 100.0],
    'direction_spread_d20_bp': [-10.0, 0.0, 10.0],
    'direction_dxy_d20_pct': [-1.0, 0.0, 1.0],
    'direction_commodity_d20_pct': [-1.0, 0.0, 1.0],
    'base_pnl_63_pct': [-2.0, 0.0, 2.0],
    'base_pnl_252_pct': [-5.0, 0.0, 5.0],
}


MACRO_PROVENANCE = {
    'K_vix_obs_date': ('K_vix', 'K_vix_d5'),
    'K_rate_obs_date': ('K_spread_bp', 'K_spread_d20_bp'),
    'K_dxy_obs_date': ('K_dxy_d20_pct',),
    'K_copper_obs_date': ('K_copper_d20_pct',),
    'K_brent_obs_date': ('K_brent_d20_pct',),
}


def _validate_entry_inputs(features, run, ledger):
    """Reject indexing ambiguity before selecting any signal-date features."""
    returns = np.asarray(run['gross_return'], float)
    if returns.ndim != 1 or len(returns) != len(features):
        raise AssertionError('BASE return run must be one-dimensional and match feature length.')
    indices = ledger.signal_idx.to_numpy(float)
    if not (np.isfinite(indices) & (indices == np.floor(indices)) &
            (indices >= 0) & (indices < len(features))).all():
        raise AssertionError('Signal indices must be finite integers inside the feature calendar.')


def _validate_macro_provenance(entry):
    """Require observation dates for finite macro values on selected rows only."""
    for observation, sources in MACRO_PROVENANCE.items():
        finite = np.zeros(len(entry), dtype=bool)
        for source in sources:
            if source in entry:
                finite |= np.isfinite(entry[source].to_numpy(float))
        if not finite.any():
            continue
        if observation not in entry or pd.to_datetime(entry[observation]).isna().to_numpy()[finite].any():
            raise AssertionError('Finite macro feature must have an observation date: ' + observation)


def entry_features(features, run, ledger, pair):
    """Read the signal-date row; BASE PnL includes only returns earned by then."""
    _validate_entry_inputs(features, run, ledger)
    indices = ledger.signal_idx.to_numpy(int)
    entry = features.iloc[indices].reset_index(drop=True).copy()
    _validate_macro_provenance(entry)
    direction = ledger.position.to_numpy(float)
    scale = entry.realized_vol_20_pct * np.sqrt(20. / 252.)
    scale = scale.where(scale.gt(0))
    entry['direction_return_20_vol'] = direction * entry.spot_return_20_pct / scale
    entry['direction_distance_ma_60_vol'] = direction * entry.distance_ma_60_pct / scale
    for source, target in [('K_spread_bp', 'direction_spread_bp'),
                           ('K_spread_d20_bp', 'direction_spread_d20_bp'),
                           ('K_dxy_d20_pct', 'direction_dxy_d20_pct')]:
        entry[target] = direction * entry[source]
    commodity = entry.K_copper_d20_pct if pair == 'AUD' else entry.K_brent_d20_pct
    entry['direction_commodity_d20_pct'] = direction * commodity if pair in ('AUD', 'CAD', 'NOK') else np.nan
    returns = pd.Series(np.asarray(run['gross_return'], float))
    for window in (63, 252):
        entry[f'base_pnl_{window}_pct'] = returns.rolling(window, min_periods=window).sum().to_numpy()[indices] * 100.
    if not pd.DatetimeIndex(entry.date).equals(pd.DatetimeIndex(ledger.signal_date)):
        raise AssertionError('Entry features must match signal dates exactly.')
    for column in (c for c in entry if c.endswith('_obs_date')):
        age = entry.date - pd.to_datetime(entry[column])
        present = age.notna()
        if not (age[present].gt(pd.Timedelta(0)) & age[present].le(pd.Timedelta(days=7))).all():
            raise AssertionError('Macro feature is not strictly previous and fresh.')
    return entry


def state_codes(values, edges):
    values = np.asarray(values, float)
    # Codes 0..3 are [-inf,e0), [e0,e1), [e1,e2), [e2,inf).
    # UNKNOWN=-1 is never implicitly turned into a rejected state.
    codes = np.searchsorted(edges, values, side='right').astype(np.int8)
    codes[~np.isfinite(values)] = -1
    return codes
