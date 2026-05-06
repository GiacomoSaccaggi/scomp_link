# -*- coding: utf-8 -*-
"""
Tests derived from examples/ to increase coverage.
Covers: pipeline save/load/predict, clustering, mixed features,
anomaly detection, time series anomaly detection, HTML report, plotly utils.
"""
import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import Lasso, ElasticNet, SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

from scomp_link import ScompLinkPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_regression_df():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'income': np.random.normal(50000, 15000, n),
        'age': np.random.randint(22, 65, n).astype(float),
        'education_years': np.random.randint(12, 20, n).astype(float),
        'salary': np.random.normal(60000, 20000, n),
    })
    df['salary'] = df['income'] * 0.3 + df['age'] * 500 + df['education_years'] * 2000
    return df


@pytest.fixture
def medium_regression_df():
    np.random.seed(42)
    n = 1500
    df = pd.DataFrame({
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'feature_3': np.random.randn(n),
        'noise_1': np.random.randn(n),
        'target': np.zeros(n),
    })
    df['target'] = 2 * df['feature_1'] + 3 * df['feature_2'] - df['feature_3'] + np.random.randn(n) * 0.5
    return df


@pytest.fixture
def mixed_features_df():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, n).astype(float),
        'income': np.random.normal(50000, 20000, n),
        'city': np.random.choice(['NYC', 'LA', 'Chicago'], n),
        'loan_amount': np.zeros(n),
    })
    df['loan_amount'] = df['income'] * 3 + df['age'] * 1000 + np.random.normal(0, 10000, n)
    return df


@pytest.fixture
def small_classification_df():
    np.random.seed(42)
    n_per = 100
    rows = []
    for cat, (mx, my) in zip(['A', 'B', 'C'], [(0, 0), (3, 3), (-3, 3)]):
        for _ in range(n_per):
            rows.append([np.random.normal(mx, 1), np.random.normal(my, 1), cat])
    return pd.DataFrame(rows, columns=['f1', 'f2', 'category'])


@pytest.fixture
def large_classification_df():
    np.random.seed(42)
    n_per = 400
    rows = []
    for cat, (ma, mi) in zip(['Low', 'Medium', 'High'], [(30, 30000), (45, 60000), (55, 100000)]):
        for _ in range(n_per):
            rows.append([np.random.normal(ma, 5), np.random.normal(mi, 5000),
                         np.random.choice(['A', 'B', 'C']), cat])
    return pd.DataFrame(rows, columns=['age', 'income', 'segment', 'risk_level'])


@pytest.fixture
def clustering_df():
    np.random.seed(42)
    rows = []
    for cx, cy in [(0, 0), (5, 5), (-5, 5), (5, -5)]:
        for _ in range(80):
            rows.append([np.random.normal(cx, 1), np.random.normal(cy, 1), np.random.randn()])
    return pd.DataFrame(rows, columns=['f1', 'f2', 'f3'])


@pytest.fixture
def anomaly_df():
    np.random.seed(42)
    normal = np.random.randn(200, 3) * [10, 50, 5] + [20, 100, 10]
    anomalies = np.random.randn(10, 3) * [50, 200, 30] + [100, 500, 50]
    data = np.vstack([normal, anomalies])
    return pd.DataFrame(data, columns=['frequency', 'duration', 'panelists'])


# ---------------------------------------------------------------------------
# Example 1 – small numerical prediction + save/load/predict
# ---------------------------------------------------------------------------

class TestNumericalSmall:

    def test_econometric_model_selected(self, small_regression_df):
        pipe = ScompLinkPipeline("Small Regression")
        pipe.import_and_clean_data(small_regression_df)
        pipe.select_variables(target_col='salary')
        pipe.choose_model("numerical_prediction", metadata={
            "only_numerical_exogenous": True,
            "all_variables_important": True
        })
        assert pipe.model_type == "Econometric Model"

    def test_save_load_predict(self, small_regression_df, tmp_path):
        pipe = ScompLinkPipeline("Small Regression")
        pipe.set_objectives(["Minimize RMSE"])
        pipe.import_and_clean_data(small_regression_df)
        pipe.select_variables(target_col='salary')
        pipe.choose_model("numerical_prediction", metadata={
            "only_numerical_exogenous": True,
            "all_variables_important": True
        })
        pipe.run_pipeline(task_type="regression")

        model_dir = str(tmp_path / "model_01")
        pipe.save_model(model_dir)

        pipe2 = ScompLinkPipeline("Loaded")
        pipe2.load_model(model_dir)

        test_data = small_regression_df.sample(3, random_state=1)[['income', 'age', 'education_years']]
        preds = pipe2.predict(test_data)
        assert len(preds) == 3


# ---------------------------------------------------------------------------
# Example 2 – medium dataset with Lasso / ElasticNet
# ---------------------------------------------------------------------------

class TestNumericalMediumLasso:

    def test_lasso_elasticnet_pipeline(self, medium_regression_df):
        pipe = ScompLinkPipeline("Medium Lasso")
        pipe.import_and_clean_data(medium_regression_df)
        pipe.select_variables(target_col='target')
        pipe.choose_model("numerical_prediction", metadata={
            "only_numerical_exogenous": True,
            "all_variables_important": False
        })
        models_to_test = {
            'Lasso': {'model': Lasso(), 'params_grid': {'alpha': [0.1, 1.0]}},
            'ElasticNet': {'model': ElasticNet(), 'params_grid': {'alpha': [0.1, 1.0], 'l1_ratio': [0.5]}},
        }
        results = pipe.run_pipeline(task_type="regression", models_to_test=models_to_test)
        assert 'optimizer_results' in results
        assert len(results['optimizer_results']) == 2


# ---------------------------------------------------------------------------
# Example 3 – mixed features (categorical + numerical)
# ---------------------------------------------------------------------------

class TestNumericalMixedFeatures:

    def test_mixed_features_pipeline(self, mixed_features_df):
        pipe = ScompLinkPipeline("Mixed Features")
        pipe.import_and_clean_data(mixed_features_df)
        pipe.select_variables(target_col='loan_amount')
        pipe.choose_model("numerical_prediction", metadata={
            "only_numerical_exogenous": False,
            "all_variables_important": True
        })
        models_to_test = {
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42, n_estimators=10),
                'params_grid': {'n_estimators': [10]}
            }
        }
        results = pipe.run_pipeline(task_type="regression", models_to_test=models_to_test)
        assert 'optimizer_results' in results

    def test_save_load_predict_mixed(self, mixed_features_df, tmp_path):
        pipe = ScompLinkPipeline("Mixed Features")
        pipe.import_and_clean_data(mixed_features_df)
        pipe.select_variables(target_col='loan_amount')
        pipe.choose_model("numerical_prediction", metadata={
            "only_numerical_exogenous": False,
            "all_variables_important": True
        })
        models_to_test = {
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42, n_estimators=10),
                'params_grid': {'n_estimators': [10]}
            }
        }
        pipe.run_pipeline(task_type="regression", models_to_test=models_to_test)
        model_dir = str(tmp_path / "model_03")
        pipe.save_model(model_dir)

        pipe2 = ScompLinkPipeline("Loaded")
        pipe2.load_model(model_dir)
        test_data = mixed_features_df.sample(3, random_state=1)[['age', 'income', 'city']]
        preds = pipe2.predict(test_data)
        assert len(preds) == 3


# ---------------------------------------------------------------------------
# Example 4 – classification small (SVC / KNeighbors / NaiveBayes)
# ---------------------------------------------------------------------------

class TestClassificationSmall:

    def test_svc_knn_nb_pipeline(self, small_classification_df):
        pipe = ScompLinkPipeline("Small Classification")
        pipe.import_and_clean_data(small_classification_df)
        pipe.select_variables(target_col='category')
        pipe.choose_model("categorical_known", metadata={
            "records_per_category": 100,
            "exogenous_type": "mixed"
        })
        models_to_test = {
            'SVC': {'model': SVC(probability=True, random_state=42), 'params_grid': {'C': [1]}},
            'KNeighbors': {'model': KNeighborsClassifier(), 'params_grid': {'n_neighbors': [3]}},
            'NaiveBayes': {'model': GaussianNB(), 'params_grid': {}},
        }
        results = pipe.run_pipeline(task_type="classification", models_to_test=models_to_test)
        assert 'optimizer_results' in results
        assert len(results['optimizer_results']) == 3

    def test_save_load_predict_classification(self, small_classification_df, tmp_path):
        pipe = ScompLinkPipeline("Small Classification")
        pipe.import_and_clean_data(small_classification_df)
        pipe.select_variables(target_col='category')
        pipe.choose_model("categorical_known", metadata={
            "records_per_category": 100,
            "exogenous_type": "mixed"
        })
        models_to_test = {
            'SVC': {'model': SVC(probability=True, random_state=42), 'params_grid': {'C': [1]}},
        }
        pipe.run_pipeline(task_type="classification", models_to_test=models_to_test)
        model_dir = str(tmp_path / "model_04")
        pipe.save_model(model_dir)

        pipe2 = ScompLinkPipeline("Loaded")
        pipe2.load_model(model_dir)
        test_data = small_classification_df.sample(3, random_state=1)[['f1', 'f2']]
        preds = pipe2.predict(test_data)
        assert len(preds) == 3


# ---------------------------------------------------------------------------
# Example 5 – classification large (SGD / GradientBoosting / RandomForest)
# ---------------------------------------------------------------------------

class TestClassificationLarge:

    def test_large_classification_pipeline(self, large_classification_df):
        pipe = ScompLinkPipeline("Large Classification")
        pipe.import_and_clean_data(large_classification_df)
        pipe.select_variables(target_col='risk_level')
        pipe.choose_model("categorical_known", metadata={
            "records_per_category": 400,
            "exogenous_type": "mixed"
        })
        models_to_test = {
            'SGD': {
                'model': SGDClassifier(random_state=42, max_iter=100),
                'params_grid': {'loss': ['hinge'], 'alpha': [0.001]}
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42, n_estimators=10),
                'params_grid': {'n_estimators': [10]}
            },
        }
        results = pipe.run_pipeline(task_type="classification", models_to_test=models_to_test)
        assert 'optimizer_results' in results


# ---------------------------------------------------------------------------
# Example 6 & 7 – clustering (known / unknown)
# ---------------------------------------------------------------------------

class TestClustering:

    def test_clustering_known_categories(self, clustering_df):
        pipe = ScompLinkPipeline("Clustering Known")
        pipe.import_and_clean_data(clustering_df)
        clustering_df['cluster'] = 0
        pipe.select_variables(target_col='cluster')
        pipe.choose_model("categorical_unknown", metadata={"categories_known": True, "n_clusters": 4})
        results = pipe.run_pipeline(task_type="clustering", n_clusters=4)
        assert results['status'] == 'success'
        assert results['n_clusters'] == 4

    def test_clustering_unknown_categories(self, clustering_df):
        pipe = ScompLinkPipeline("Clustering Unknown")
        pipe.import_and_clean_data(clustering_df)
        clustering_df['cluster'] = 0
        pipe.select_variables(target_col='cluster')
        pipe.choose_model("categorical_unknown", metadata={"categories_known": False})
        results = pipe.run_pipeline(task_type="clustering")
        assert results['status'] == 'success'
        assert results['n_clusters'] >= 1


# ---------------------------------------------------------------------------
# Example 8 – very large dataset (SGDRegressor)
# ---------------------------------------------------------------------------

class TestNumericalVeryLarge:

    def test_sgd_regressor_pipeline(self):
        np.random.seed(42)
        n = 500  # keep small for test speed
        df = pd.DataFrame({
            'f1': np.random.randn(n),
            'f2': np.random.randn(n),
            'f3': np.random.randn(n),
            'target': np.zeros(n),
        })
        df['target'] = 1.5 * df['f1'] + 2 * df['f2'] + np.random.randn(n) * 0.3

        pipe = ScompLinkPipeline("SGD Large")
        pipe.import_and_clean_data(df)
        pipe.select_variables(target_col='target')
        pipe.choose_model("numerical_prediction", metadata={
            "only_numerical_exogenous": True,
            "all_variables_important": True
        })
        models_to_test = {
            'SGDRegressor': {
                'model': SGDRegressor(random_state=42, max_iter=100),
                'params_grid': {'loss': ['squared_error'], 'alpha': [0.0001]}
            }
        }
        results = pipe.run_pipeline(task_type="regression", models_to_test=models_to_test)
        assert 'optimizer_results' in results


# ---------------------------------------------------------------------------
# Example 15 – AnomalyDetector (iforest + lof only, no deep learning)
# ---------------------------------------------------------------------------

class TestAnomalyDetector:

    def test_fit_predict_iforest_lof(self, anomaly_df):
        from scomp_link import AnomalyDetector
        detector = AnomalyDetector(
            contamination=0.05,
            methods=['iforest', 'lof'],
            consensus_threshold=2,
            verbose=False,
        )
        results = detector.fit_predict(anomaly_df, features=['frequency', 'duration', 'panelists'])
        assert 'data' in results
        assert 'comparison' in results
        assert 'is_anomaly' in results['data'].columns
        assert len(results['comparison']) == 3  # 2 methods + consensus row

    def test_report_no_groupby(self, anomaly_df):
        from scomp_link import AnomalyDetector
        detector = AnomalyDetector(contamination=0.05, methods=['iforest', 'lof'],
                                   consensus_threshold=1, verbose=False)
        detector.fit_predict(anomaly_df, features=['frequency', 'duration', 'panelists'])
        report = detector.report()
        assert isinstance(report, pd.DataFrame)

    def test_report_before_fit_raises(self):
        from scomp_link import AnomalyDetector
        detector = AnomalyDetector(methods=['iforest'], verbose=False)
        with pytest.raises(ValueError, match="No results"):
            detector.report()

    def test_unknown_method_raises(self, anomaly_df):
        from scomp_link import AnomalyDetector
        detector = AnomalyDetector(methods=['unknown_method'], verbose=False)
        with pytest.raises(ValueError, match="Unknown method"):
            detector.fit_predict(anomaly_df, features=['frequency', 'duration', 'panelists'])


# ---------------------------------------------------------------------------
# Example 16 – TimeSeriesAnomalyDetector (statistical methods only)
# ---------------------------------------------------------------------------

class TestTimeSeriesAnomalyDetector:

    @pytest.fixture
    def ts_data(self):
        np.random.seed(42)
        t = np.linspace(0, 10 * np.pi, 500)
        normal = 20 + 5 * np.sin(t) + np.random.randn(500) * 0.5
        test = normal.copy()
        test[100:110] = 50
        test[300] = -10
        return normal, test

    def test_fit_detect_moving_avg(self, ts_data):
        from scomp_link import TimeSeriesAnomalyDetector
        train, test = ts_data
        detector = TimeSeriesAnomalyDetector(
            methods=['moving_avg'],
            window_size=20,
            n_sigma=3.0,
            verbose=False,
        )
        detector.fit(train)
        results = detector.detect(test)
        assert 'anomalies' in results
        assert 'methods' in results
        assert 'moving_avg' in results['methods']
        assert results['anomalies'].dtype == bool

    def test_fit_detect_moving_median(self, ts_data):
        from scomp_link import TimeSeriesAnomalyDetector
        train, test = ts_data
        detector = TimeSeriesAnomalyDetector(
            methods=['moving_median'],
            window_size=20,
            n_sigma=3.0,
            verbose=False,
        )
        detector.fit(train)
        results = detector.detect(test)
        assert results['anomalies'].sum() >= 0

    def test_fit_detect_arima(self, ts_data):
        pytest.importorskip('statsmodels')
        from scomp_link import TimeSeriesAnomalyDetector
        train, test = ts_data
        detector = TimeSeriesAnomalyDetector(
            methods=['arima'],
            arima_order=(2, 1, 0),
            verbose=False,
        )
        detector.fit(train)
        results = detector.detect(test)
        assert 'arima' in results['methods']

    def test_consensus_score_shape(self, ts_data):
        from scomp_link import TimeSeriesAnomalyDetector
        train, test = ts_data
        detector = TimeSeriesAnomalyDetector(
            methods=['moving_avg', 'moving_median'],
            window_size=20,
            verbose=False,
        )
        detector.fit(train)
        results = detector.detect(test)
        assert len(results['consensus_score']) == len(test)

    def test_unknown_method_raises(self, ts_data):
        from scomp_link import TimeSeriesAnomalyDetector
        train, test = ts_data
        detector = TimeSeriesAnomalyDetector(methods=['bad_method'], verbose=False)
        detector.fit(train)
        with pytest.raises(ValueError, match="Unknown method"):
            detector.detect(test)


# ---------------------------------------------------------------------------
# Example 17 – ScompLinkHTMLReport
# ---------------------------------------------------------------------------

class TestHTMLReport:

    def test_basic_report_saves(self, tmp_path):
        from scomp_link.utils.report_html import ScompLinkHTMLReport
        report = ScompLinkHTMLReport('Test Report')
        report.add_title('Hello')
        report.add_text('Some text')
        out = str(tmp_path / 'report.html')
        report.save_html(out)
        assert os.path.exists(out)
        content = open(out, encoding='utf-8').read()
        assert 'Hello' in content

    def test_add_dataframe(self, tmp_path):
        from scomp_link.utils.report_html import ScompLinkHTMLReport
        report = ScompLinkHTMLReport('DF Report')
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        report.add_dataframe(df, 'my_table')
        out = str(tmp_path / 'df_report.html')
        report.save_html(out)
        content = open(out, encoding='utf-8').read()
        assert 'my_table' in content

    def test_add_plotly_graph(self, tmp_path):
        import plotly.express as px
        from scomp_link.utils.report_html import ScompLinkHTMLReport
        report = ScompLinkHTMLReport('Graph Report')
        fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6])
        report.add_graph_to_report(fig, 'Scatter Plot')
        out = str(tmp_path / 'graph_report.html')
        report.save_html(out)
        assert os.path.exists(out)

    def test_sections_open_close(self, tmp_path):
        from scomp_link.utils.report_html import ScompLinkHTMLReport
        report = ScompLinkHTMLReport('Section Report')
        report.open_section('Section 1')
        report.add_title('Inside section')
        report.close_section()
        out = str(tmp_path / 'section_report.html')
        report.save_html(out)
        content = open(out, encoding='utf-8').read()
        assert 'Section 1' in content

    def test_add_many_plots_single_key(self, tmp_path):
        import plotly.express as px
        from scomp_link.utils.report_html import ScompLinkHTMLReport
        report = ScompLinkHTMLReport('Multi Plot')
        figs = {'Option A': px.scatter(x=[1], y=[1]), 'Option B': px.scatter(x=[2], y=[2])}
        report.add_many_plots_with_selection_box_to_report(figs, 'My Combo')
        out = str(tmp_path / 'multi_report.html')
        report.save_html(out)
        assert os.path.exists(out)

    def test_add_many_plots_tuple_key(self, tmp_path):
        import plotly.express as px
        from scomp_link.utils.report_html import ScompLinkHTMLReport
        report = ScompLinkHTMLReport('Tuple Plot')
        figs = {
            ('First', 'Blue'): px.scatter(x=[1], y=[1]),
            ('Second', 'Blue'): px.scatter(x=[2], y=[2]),
        }
        report.add_many_plots_with_selection_box_to_report(figs, 'Tuple Combo', labels=['Number', 'Color'])
        out = str(tmp_path / 'tuple_report.html')
        report.save_html(out)
        assert os.path.exists(out)

    def test_add_matplotlib_graph(self, tmp_path):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scomp_link.utils.report_html import ScompLinkHTMLReport
        report = ScompLinkHTMLReport('MPL Report')
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        report.add_matplotlib_graph_to_report(fig, 'MPL Graph')
        plt.close(fig)
        out = str(tmp_path / 'mpl_report.html')
        report.save_html(out)
        content = open(out, encoding='utf-8').read()
        assert 'MPL Graph' in content


# ---------------------------------------------------------------------------
# Example 17 – plotly_utils
# ---------------------------------------------------------------------------

class TestPlotlyUtils:

    def test_histogram(self):
        from scomp_link.utils.plotly_utils import histogram
        x = np.random.normal(45, 3, 100)
        fig = histogram(x, 'Age')
        assert fig is not None

    def test_multiple_histograms(self):
        from scomp_link.utils.plotly_utils import multiple_histograms
        x = np.random.normal(85, 3, 90)
        cats = ['A', 'B', 'C'] * 30
        fig = multiple_histograms(x, cats, 'Category')
        assert fig is not None

    def test_multiple_histograms_too_many_categories(self):
        from scomp_link.utils.plotly_utils import multiple_histograms
        x = np.random.randn(110)
        cats = [str(i) for i in range(11)] * 10
        fig = multiple_histograms(x, cats, 'Many')
        assert fig is None

    def test_barchart(self):
        from scomp_link.utils.plotly_utils import barchart
        fig = barchart(
            categories=['A', 'B', 'C'],
            metric_values_list=[[10, 20, 30]],
            y_axis_titles=['Value'],
        )
        assert fig is not None

    def test_barchart_with_line(self):
        from scomp_link.utils.plotly_utils import barchart
        fig = barchart(
            categories=['A', 'B', 'C'],
            metric_values_list=[[10, 20, 30]],
            metric_values_line_list=[[15, 25, 35]],
            y_axis_titles='Value',
            y_line_axis_titles='Target',
        )
        assert fig is not None

    def test_linechart(self):
        from scomp_link.utils.plotly_utils import linechart
        dates = ['2024-01-01', '2024-02-01', '2024-03-01']
        fig = linechart(date_list=dates, lines=[[10, 20, 30]], y_labels=['Series 1'])
        assert fig is not None

    def test_area_chart(self):
        from scomp_link.utils.plotly_utils import area_chart
        dates = ['2024-01-01', '2024-02-01', '2024-03-01']
        fig = area_chart(date_list=dates, lines=[[100, 200, 150]], y_labels=['Views'])
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
