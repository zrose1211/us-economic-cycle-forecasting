"""Reproducible, leakage-aware one-quarter-ahead U.S. unemployment modeling.

Run: python -m src.pipeline
All input observations were sourced from the statsmodels historical macrodata example.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'us_macro_1959_2009.csv'
OUT = ROOT / 'reports'
FEATURE_COLUMNS = [
    'unemp', 'unemp_change_prev', 'gdp_growth_qoq', 'cons_growth_qoq',
    'investment_growth_qoq', 'infl', 'tbilrate', 'realint',
]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    raw = pd.read_csv(path)
    expected = {'year', 'quarter', 'realgdp', 'realcons', 'realinv', 'unemp', 'infl', 'tbilrate', 'realint'}
    missing = expected - set(raw.columns)
    if missing:
        raise ValueError(f'Missing input columns: {sorted(missing)}')
    raw = raw.sort_values(['year', 'quarter']).reset_index(drop=True)
    if raw[['year', 'quarter']].duplicated().any():
        raise ValueError('Duplicate quarters in input')
    return raw


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy().sort_values(['year', 'quarter']).reset_index(drop=True)
    frame['period'] = pd.PeriodIndex.from_fields(year=frame['year'].astype(int), quarter=frame['quarter'].astype(int), freq='Q').astype(str)
    frame['gdp_growth_qoq'] = frame['realgdp'].pct_change(fill_method=None) * 100
    frame['cons_growth_qoq'] = frame['realcons'].pct_change(fill_method=None) * 100
    frame['investment_growth_qoq'] = frame['realinv'].pct_change(fill_method=None) * 100
    frame['unemp_change_prev'] = frame['unemp'].diff()
    # Each row uses *only* information from period t, predicting unemployment at t+1.
    frame['target_next_unemp'] = frame['unemp'].shift(-1)
    frame = frame.dropna(subset=FEATURE_COLUMNS + ['target_next_unemp']).reset_index(drop=True)
    return frame


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        'mae_percentage_points': round(float(mean_absolute_error(y_true, y_pred)), 4),
        'rmse_percentage_points': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
    }


def tune_alpha(x: pd.DataFrame, y: pd.Series) -> tuple[float, list[dict]]:
    cv = TimeSeriesSplit(n_splits=5, test_size=12)
    rows = []
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        fold_maes = []
        for train_idx, val_idx in cv.split(x):
            model = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=alpha))])
            model.fit(x.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(x.iloc[val_idx])
            fold_maes.append(float(mean_absolute_error(y.iloc[val_idx], preds)))
        rows.append({'alpha': alpha, 'mean_cv_mae_pp': round(float(np.mean(fold_maes)), 4)})
    rows.sort(key=lambda r: (r['mean_cv_mae_pp'], r['alpha']))
    return float(rows[0]['alpha']), rows


def export_sql(raw: pd.DataFrame, dataset: pd.DataFrame) -> None:
    db_path = OUT / 'macro_analysis.sqlite'
    with sqlite3.connect(db_path) as conn:
        raw.to_sql('quarterly_indicators', conn, if_exists='replace', index=False)
        dataset.to_sql('forecast_features', conn, if_exists='replace', index=False)
        query = (ROOT / 'sql' / 'analysis.sql').read_text()
        # Multi-query file separated by semicolon; each query has a purpose-built table.
        for idx, statement in enumerate(query.split(';'), start=1):
            statement = statement.strip()
            if statement:
                pd.read_sql_query(statement, conn).to_csv(OUT / f'sql_result_{idx}.csv', index=False)


def make_plots(test: pd.DataFrame, predictions: np.ndarray, baseline: np.ndarray, model: Pipeline) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    idx = np.arange(len(test))
    ax.plot(idx, test['target_next_unemp'].to_numpy(), label='Observed next-quarter unemployment', color='#163b6c', linewidth=2.3)
    ax.plot(idx, predictions, label='Ridge model', color='#e47734', linewidth=1.8)
    ax.plot(idx, baseline, label='Persistence baseline', color='#73849a', linewidth=1.3, alpha=.8)
    step = max(1, len(idx) // 9)
    ax.set_xticks(idx[::step], test['period'].to_numpy()[::step], rotation=35)
    ax.set_ylabel('Unemployment rate (%)')
    ax.set_xlabel('Predictor quarter (target is next quarter)')
    ax.set_title('Out-of-time evaluation: 1999-2009', loc='left', fontsize=15, weight='bold')
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / 'unemployment_forecast.png', dpi=180)
    plt.close(fig)

    coefficients = model.named_steps['ridge'].coef_
    order = np.argsort(abs(coefficients))
    fig, ax = plt.subplots(figsize=(10.3, 4.8))
    ax.barh(np.array(FEATURE_COLUMNS)[order], coefficients[order], color=['#d87b58' if c < 0 else '#2c8392' for c in coefficients[order]])
    ax.axvline(0, color='#222b36', linewidth=.8)
    ax.set_xlabel('Ridge coefficient (per 1 standardized feature unit)')
    ax.set_title('Model associations, not causal effects', loc='left', fontsize=15, weight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / 'standardized_coefficients.png', dpi=180)
    plt.close(fig)


def main() -> dict:
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / 'figures').mkdir(exist_ok=True, parents=True)
    raw = load_data()
    data = make_features(raw)
    train = data.loc[data['year'] < 1999].copy()
    test = data.loc[data['year'] >= 1999].copy()
    assert len(train) > 100 and len(test) > 30, 'Unexpected split sizes'
    x_train, y_train = train[FEATURE_COLUMNS], train['target_next_unemp']
    x_test, y_test = test[FEATURE_COLUMNS], test['target_next_unemp']
    alpha, trials = tune_alpha(x_train, y_train)
    model = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=alpha))])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    baseline = test['unemp'].to_numpy()  # next quarter = this quarter
    model_results = evaluate(y_test.to_numpy(), predictions)
    base_results = evaluate(y_test.to_numpy(), baseline)
    results = {
        'data': 'statsmodels macrodata; quarterly 1959Q1 through 2009Q3',
        'task': 'Predict next-quarter unemployment rate from current and prior-quarter indicators',
        'train_period': f"{train.iloc[0]['period']} to {train.iloc[-1]['period']}",
        'test_predictor_period': f"{test.iloc[0]['period']} to {test.iloc[-1]['period']}",
        'train_rows': len(train), 'test_rows': len(test),
        'tuned_ridge_alpha': alpha, 'cv_trials': trials,
        'ridge': model_results, 'persistence_baseline': base_results,
        'caveats': [
            'Historic dataset ends in 2009 and is not a current live forecast.',
            'Economic indicators can be revised after their original release; evaluation uses revised historical values.',
            'Small sample and structural changes limit generalizability; associations are not causal.',
        ],
    }
    (OUT / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    pred_df = test[['period', 'unemp', 'target_next_unemp']].copy()
    pred_df['ridge_pred'] = predictions
    pred_df['persistence_pred'] = baseline
    pred_df.to_csv(OUT / 'test_predictions.csv', index=False)
    export_sql(raw, data)
    make_plots(test, predictions, baseline, model)
    print(json.dumps(results, indent=2))
    return results


if __name__ == '__main__':
    main()
