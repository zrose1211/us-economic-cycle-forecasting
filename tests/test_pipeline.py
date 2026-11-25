import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from src.pipeline import FEATURE_COLUMNS, load_data, make_features, tune_alpha, evaluate


def test_feature_labels_reference_next_quarter():
    raw = load_data()
    features = make_features(raw)
    assert len(raw) == 203
    assert len(features) == 201
    assert not features[FEATURE_COLUMNS + ['target_next_unemp']].isna().any().any()
    current = features.iloc[0]
    next_row = raw.loc[(raw['year'] == current['year']) & (raw['quarter'] == current['quarter'])].index[0] + 1
    assert np.isclose(current['target_next_unemp'], raw.iloc[next_row]['unemp'])
    assert 'target_next_unemp' not in FEATURE_COLUMNS


def test_chronological_cross_validation_and_pipeline():
    feat = make_features(load_data())
    train = feat.loc[feat['year'] < 1999]
    x = train[FEATURE_COLUMNS]
    y = train['target_next_unemp']
    for a, b in TimeSeriesSplit(n_splits=5, test_size=12).split(x):
        assert max(a) < min(b)
    chosen_alpha, trials = tune_alpha(x, y)
    assert len(trials) == 5 and chosen_alpha in {0.01, 0.1, 1.0, 10.0, 100.0}


def test_evaluate_error_units():
    x = evaluate(np.array([4.0, 5.0]), np.array([4.5, 4.5]))
    assert x['mae_percentage_points'] == 0.5
    assert x['rmse_percentage_points'] == 0.5
