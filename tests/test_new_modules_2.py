# -*- coding: utf-8 -*-
"""
Test suite for: FeatureEngineer, TimeSeriesForecaster, FairnessMetrics, DataQualityReport
"""
import pytest
import numpy as np
import pandas as pd
import os
import tempfile


# ===================== FEATURE ENGINEER =====================

class TestFeatureEngineer:

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = pd.DataFrame({
            'income': np.random.exponential(50000, 200),
            'age': np.random.normal(35, 10, 200),
            'score': np.random.exponential(100, 200),
            'city': np.random.choice(['NYC', 'LA', 'CHI', 'HOU', 'PHX', 'PHI',
                                       'SA', 'SD', 'DAL', 'SJ', 'AUS', 'JAX'], 200),
        })
        y = 0.5 * X['income'] + 100 * X['age']
        return X, y

    def test_log_transform_skewed(self, sample_data):
        from scomp_link import FeatureEngineer
        X, y = sample_data
        fe = FeatureEngineer(interactions=False, log_transform=True, date_features=False,
                             target_encode=False, auto_bin=False)
        X_eng = fe.fit_transform(X, y)
        assert 'income_log' in X_eng.columns
        assert 'score_log' in X_eng.columns
        assert 'age_log' not in X_eng.columns  # not skewed enough

    def test_interactions(self, sample_data):
        from scomp_link import FeatureEngineer
        X, y = sample_data
        X_num = X[['income', 'age', 'score']]
        fe = FeatureEngineer(interactions=True, log_transform=False, date_features=False,
                             target_encode=False, auto_bin=False)
        X_eng = fe.fit_transform(X_num, y)
        assert 'income_x_age' in X_eng.columns
        assert 'income_x_score' in X_eng.columns

    def test_target_encoding(self, sample_data):
        from scomp_link import FeatureEngineer
        X, y = sample_data
        fe = FeatureEngineer(interactions=False, log_transform=False, date_features=False,
                             target_encode=True, target_encode_threshold=5, auto_bin=False)
        X_eng = fe.fit_transform(X, y)
        assert 'city_target_enc' in X_eng.columns
        assert 'city' not in X_eng.columns

    def test_date_features(self):
        from scomp_link import FeatureEngineer
        X = pd.DataFrame({
            'value': np.random.randn(50),
            'date': pd.date_range('2020-01-01', periods=50, freq='D'),
        })
        fe = FeatureEngineer(interactions=False, log_transform=False, date_features=True,
                             target_encode=False, auto_bin=False)
        X_eng = fe.fit_transform(X)
        assert 'date_month' in X_eng.columns
        assert 'date_day_of_week' in X_eng.columns
        assert 'date_is_weekend' in X_eng.columns
        assert 'date' not in X_eng.columns

    def test_auto_bin(self, sample_data):
        from scomp_link import FeatureEngineer
        X, y = sample_data
        X_num = X[['income', 'age']]
        fe = FeatureEngineer(interactions=False, log_transform=False, date_features=False,
                             target_encode=False, auto_bin=True, n_bins=4)
        X_eng = fe.fit_transform(X_num, y)
        assert 'income_bin' in X_eng.columns
        assert 'age_bin' in X_eng.columns

    def test_sklearn_compatible(self, sample_data):
        from scomp_link import FeatureEngineer
        X, y = sample_data
        X_num = X[['income', 'age']]
        fe = FeatureEngineer(interactions=False, log_transform=True, auto_bin=False,
                             date_features=False, target_encode=False)
        fe.fit(X_num, y)
        X_out = fe.transform(X_num)
        assert isinstance(X_out, pd.DataFrame)
        assert len(X_out) == len(X_num)


# ===================== TIME SERIES FORECASTER =====================

class TestTimeSeriesForecaster:

    @pytest.fixture
    def trend_series(self):
        np.random.seed(42)
        t = np.arange(120)
        return pd.Series(50 + 0.5 * t + np.random.randn(120) * 2)

    @pytest.fixture
    def seasonal_series(self):
        np.random.seed(42)
        t = np.arange(120)
        return pd.Series(50 + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(120) * 2)

    def test_arima_fit_predict(self, trend_series):
        from scomp_link import TimeSeriesForecaster
        fc = TimeSeriesForecaster(method='arima', horizon=10)
        fc.fit(trend_series[:100])
        pred = fc.predict()
        assert len(pred) == 10
        assert pred.mean() > 0

    def test_exp_smoothing_fit_predict(self, seasonal_series):
        from scomp_link import TimeSeriesForecaster
        fc = TimeSeriesForecaster(method='exp_smoothing', horizon=12, seasonal_period=12)
        fc.fit(seasonal_series[:100])
        pred = fc.predict()
        assert len(pred) == 12

    def test_predict_before_fit_raises(self):
        from scomp_link import TimeSeriesForecaster
        fc = TimeSeriesForecaster(method='arima')
        with pytest.raises(ValueError):
            fc.predict()

    def test_predict_with_ci(self, trend_series):
        from scomp_link import TimeSeriesForecaster
        fc = TimeSeriesForecaster(method='arima', horizon=5)
        fc.fit(trend_series[:100])
        ci = fc.predict_with_ci()
        assert 'forecast' in ci.columns
        assert 'lower' in ci.columns
        assert 'upper' in ci.columns
        assert (ci['upper'] >= ci['forecast']).all()
        assert (ci['lower'] <= ci['forecast']).all()

    def test_walk_forward_cv(self, trend_series):
        from scomp_link import TimeSeriesForecaster
        fc = TimeSeriesForecaster(method='arima', horizon=10)
        results = fc.walk_forward_cv(trend_series, n_splits=3, horizon=10)
        assert 'mean_mae' in results
        assert 'mean_rmse' in results
        assert len(results['folds']) == 3

    def test_auto_method(self, trend_series):
        from scomp_link import TimeSeriesForecaster
        fc = TimeSeriesForecaster(method='auto', horizon=5)
        fc.fit(trend_series[:100])
        pred = fc.predict()
        assert len(pred) == 5

    def test_plot_forecast(self, trend_series):
        from scomp_link import TimeSeriesForecaster
        fc = TimeSeriesForecaster(method='arima', horizon=10)
        fc.fit(trend_series[:100])
        fig = fc.plot_forecast()
        assert hasattr(fig, 'to_json')


# ===================== FAIRNESS METRICS =====================

class TestFairnessMetrics:

    @pytest.fixture
    def fair_data(self):
        np.random.seed(42)
        n = 500
        y_true = np.random.binomial(1, 0.5, n)
        y_pred = y_true.copy()
        group = np.random.choice(['A', 'B'], n)
        return y_true, y_pred, group

    @pytest.fixture
    def biased_data(self):
        np.random.seed(42)
        n = 500
        group = np.random.choice(['privileged', 'unprivileged'], n)
        y_true = np.random.binomial(1, 0.5, n)
        y_pred = y_true.copy()
        # Bias: reject qualified unprivileged
        mask = (group == 'unprivileged') & (y_true == 1)
        flip = np.where(mask)[0][:80]
        y_pred[flip] = 0
        return y_true, y_pred, group

    def test_demographic_parity_fair(self, fair_data):
        from scomp_link import FairnessMetrics
        fm = FairnessMetrics(*fair_data)
        dp = fm.demographic_parity()
        assert dp['fair'] == True
        assert dp['dp_ratio'] > 0.8

    def test_demographic_parity_biased(self, biased_data):
        from scomp_link import FairnessMetrics
        fm = FairnessMetrics(*biased_data)
        dp = fm.demographic_parity()
        assert dp['dp_ratio'] < 0.9  # biased model

    def test_disparate_impact_four_fifths(self, biased_data):
        from scomp_link import FairnessMetrics
        fm = FairnessMetrics(*biased_data)
        di = fm.disparate_impact()
        assert 'di_ratio' in di
        assert 'four_fifths_rule' in di

    def test_equalized_odds(self, fair_data):
        from scomp_link import FairnessMetrics
        fm = FairnessMetrics(*fair_data)
        eo = fm.equalized_odds()
        assert eo['tpr_diff'] < 0.15
        assert 'tpr' in eo
        assert 'fpr' in eo

    def test_equal_opportunity(self, biased_data):
        from scomp_link import FairnessMetrics
        fm = FairnessMetrics(*biased_data)
        eop = fm.equal_opportunity()
        assert eop['tpr_diff'] > 0  # biased

    def test_compute_all(self, fair_data):
        from scomp_link import FairnessMetrics
        fm = FairnessMetrics(*fair_data)
        report = fm.compute_all()
        assert len(report) == 4
        assert 'demographic_parity' in report
        assert 'equalized_odds' in report

    def test_summary_dataframe(self, biased_data):
        from scomp_link import FairnessMetrics
        fm = FairnessMetrics(*biased_data)
        summary = fm.summary()
        assert isinstance(summary, pd.DataFrame)
        assert 'metric' in summary.columns
        assert len(summary) == 5

    def test_plot_returns_figure(self, biased_data):
        from scomp_link import FairnessMetrics
        fm = FairnessMetrics(*biased_data)
        fig = fm.plot_fairness_report()
        assert hasattr(fig, 'to_json')


# ===================== DATA QUALITY REPORT =====================

class TestDataQualityReport:

    @pytest.fixture
    def messy_data(self):
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'id': range(n),
            'value': np.random.randn(n),
            'correlated': np.random.randn(n),
            'category': np.random.choice(['A', 'B'], n),
            'constant': 'same',
            'mostly_null': np.where(np.random.rand(n) > 0.1, np.nan, 1.0),
        })
        df['nearly_same'] = df['value'] + np.random.randn(n) * 0.001  # high correlation
        return df

    def test_generate_report(self, messy_data):
        from scomp_link import DataQualityReport
        dqr = DataQualityReport(messy_data)
        report = dqr.generate()
        assert 'overview' in report
        assert 'missing' in report
        assert 'constants' in report
        assert report['overview']['n_rows'] == 200

    def test_detects_constants(self, messy_data):
        from scomp_link import DataQualityReport
        dqr = DataQualityReport(messy_data)
        report = dqr.generate()
        assert 'constant' in report['constants']

    def test_detects_missing(self, messy_data):
        from scomp_link import DataQualityReport
        dqr = DataQualityReport(messy_data)
        report = dqr.generate()
        assert len(report['missing']) > 0
        assert 'mostly_null' in report['missing']['column'].values

    def test_duplicates(self):
        from scomp_link import DataQualityReport
        df = pd.DataFrame({'a': [1, 1, 2, 2, 3], 'b': [10, 10, 20, 20, 30]})
        dqr = DataQualityReport(df)
        report = dqr.generate()
        assert report['duplicates']['n_duplicates'] == 2

    def test_high_correlations(self, messy_data):
        from scomp_link import DataQualityReport
        dqr = DataQualityReport(messy_data)
        report = dqr.generate()
        corr_df = report['correlations']
        assert len(corr_df) > 0  # value and nearly_same are highly correlated

    def test_cardinality(self, messy_data):
        from scomp_link import DataQualityReport
        dqr = DataQualityReport(messy_data)
        report = dqr.generate()
        card = report['cardinality']
        id_row = card[card['column'] == 'id'].iloc[0]
        assert 'near-unique' in id_row['flag']

    def test_save_html(self, messy_data):
        from scomp_link import DataQualityReport
        dqr = DataQualityReport(messy_data)
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = f.name
        try:
            dqr.save_html(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 500
        finally:
            os.unlink(path)
