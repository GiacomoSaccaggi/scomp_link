# -*- coding: utf-8 -*-
"""
███████╗ ██████╗ ██████╗ ███████╗ ██████╗ █████╗ ███████╗████████╗███████╗██████╗ 
██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔══██╗
█████╗  ██║   ██║██████╔╝█████╗  ██║     ███████║███████╗   ██║   █████╗  ██████╔╝
██╔══╝  ██║   ██║██╔══██╗██╔══╝  ██║     ██╔══██║╚════██║   ██║   ██╔══╝  ██╔══██╗
██║     ╚██████╔╝██║  ██║███████╗╚██████╗██║  ██║███████║   ██║   ███████╗██║  ██║
╚═╝      ╚═════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Union, Tuple

from scomp_link.utils.logger import get_logger
logger = get_logger(__name__)
from scomp_link.utils.decorators import timer, validate_args



class TimeSeriesForecaster:
    """
    Unified time series forecasting with multiple methods and walk-forward validation.

    Dependencies: statsmodels, numpy, pandas, plotly

    PARAMETERS:
     1. method: forecasting method ('arima', 'sarima', 'exp_smoothing', 'auto')
     2. horizon: number of steps to forecast (default 10)
     3. seasonal_period: seasonality period for SARIMA/ETS (default None = auto-detect)
     4. order: ARIMA order (p,d,q) — if None, auto-selected
     5. seasonal_order: SARIMA seasonal order (P,D,Q,s) — if None, auto-selected

    Usage example:
        forecaster = TimeSeriesForecaster(method='arima', horizon=30)
        forecaster.fit(train_series)
        forecast = forecaster.predict()
        metrics = forecaster.walk_forward_cv(full_series, n_splits=5)
    """

    def __init__(self, method: str = "auto", horizon: int = 10,
                 seasonal_period: Optional[int] = None,
                 order: Optional[Tuple[int, int, int]] = None,
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None):
        self.method = method
        self.horizon = horizon
        self.seasonal_period = seasonal_period
        self.order = order
        self.seasonal_order = seasonal_order
        self._model = None
        self._fitted = None
        self._train_series = None

    @timer
    def fit(self, series: Union[pd.Series, np.ndarray], **kwargs) -> "TimeSeriesForecaster":
        """Fit the forecasting model on a time series."""
        import statsmodels.api as sm

        if isinstance(series, np.ndarray):
            series = pd.Series(series)
        self._train_series = series.copy()

        method = self._resolve_method(series) if self.method == "auto" else self.method
        logger.info(f"🔬 TimeSeriesForecaster: fitting with method='{method}'...")

        if method == "arima":
            order = self.order or self._auto_arima_order(series)
            self._model = sm.tsa.ARIMA(series, order=order)
            self._fitted = self._model.fit()
            logger.info(f"  ✅ ARIMA{order} fitted (AIC={self._fitted.aic:.2f})")

        elif method == "sarima":
            order = self.order or (1, 1, 1)
            s = self.seasonal_period or self._detect_seasonality(series)
            seasonal_order = self.seasonal_order or (1, 1, 1, s)
            self._model = sm.tsa.SARIMAX(series, order=order, seasonal_order=seasonal_order)
            self._fitted = self._model.fit(disp=False)
            logger.info(f"  ✅ SARIMA{order}x{seasonal_order} fitted (AIC={self._fitted.aic:.2f})")

        elif method == "exp_smoothing":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            s = self.seasonal_period or self._detect_seasonality(series)
            seasonal = 'add' if s and s > 1 else None
            self._model = ExponentialSmoothing(series, trend='add', seasonal=seasonal,
                                               seasonal_periods=s if seasonal else None)
            self._fitted = self._model.fit(optimized=True)
            logger.info(f"  ✅ ExponentialSmoothing fitted (seasonal_period={s})")

        return self

    def predict(self, steps: Optional[int] = None) -> pd.Series:
        """Forecast future values."""
        if self._fitted is None:
            raise ValueError("Call fit() first.")
        h = steps or self.horizon
        forecast = self._fitted.forecast(h)
        return pd.Series(forecast, name='forecast')

    def predict_with_ci(self, steps: Optional[int] = None, alpha: float = 0.05) -> pd.DataFrame:
        """Forecast with confidence intervals."""
        if self._fitted is None:
            raise ValueError("Call fit() first.")
        h = steps or self.horizon
        pred = self._fitted.get_forecast(h)
        ci = pred.conf_int(alpha=alpha)
        df = pd.DataFrame({
            'forecast': pred.predicted_mean.values,
            'lower': ci.iloc[:, 0].values,
            'upper': ci.iloc[:, 1].values,
        })
        return df

    @timer
    def walk_forward_cv(self, series: Union[pd.Series, np.ndarray],
                        n_splits: int = 5, horizon: Optional[int] = None) -> Dict:
        """
        Walk-forward cross-validation (expanding window).
        Returns dict with per-fold and aggregate metrics.
        """
        if isinstance(series, np.ndarray):
            series = pd.Series(series)
        h = horizon or self.horizon
        n = len(series)
        min_train = n - (n_splits * h)
        if min_train < 10:
            min_train = max(10, n // 2)

        fold_metrics = []
        for i in range(n_splits):
            train_end = min_train + i * h
            test_end = train_end + h
            if test_end > n:
                break

            train = series.iloc[:train_end]
            test = series.iloc[train_end:test_end]

            self.fit(train)
            forecast = self.predict(steps=len(test))

            mae = np.abs(test.values - forecast.values).mean()
            rmse = np.sqrt(((test.values - forecast.values) ** 2).mean())
            mape = (np.abs((test.values - forecast.values) / (test.values + 1e-8))).mean() * 100

            fold_metrics.append({'fold': i + 1, 'mae': mae, 'rmse': rmse, 'mape': mape})

        results_df = pd.DataFrame(fold_metrics)
        logger.info(f"  ✅ Walk-forward CV: {len(fold_metrics)} folds")
        logger.info(f"     Mean MAE={results_df['mae'].mean():.4f}, RMSE={results_df['rmse'].mean():.4f}")
        return {
            'folds': results_df,
            'mean_mae': float(results_df['mae'].mean()),
            'mean_rmse': float(results_df['rmse'].mean()),
            'mean_mape': float(results_df['mape'].mean()),
        }

    def plot_forecast(self, forecast: Optional[pd.Series] = None,
                      ci: Optional[pd.DataFrame] = None) -> "plotly.graph_objects.Figure":
        """Plot historical data + forecast with optional confidence interval."""
        import plotly.graph_objects as go
        if forecast is None:
            forecast = self.predict()

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=self._train_series.values, name='Historical', mode='lines'))

        x_forecast = list(range(len(self._train_series), len(self._train_series) + len(forecast)))
        fig.add_trace(go.Scatter(x=x_forecast, y=forecast.values, name='Forecast',
                                 mode='lines', line=dict(dash='dash', color='red')))

        if ci is not None:
            fig.add_trace(go.Scatter(x=x_forecast + x_forecast[::-1],
                                     y=list(ci['upper'].values) + list(ci['lower'].values[::-1]),
                                     fill='toself', fillcolor='rgba(255,0,0,0.1)',
                                     line=dict(color='rgba(255,0,0,0)'), name='CI'))

        fig.update_layout(title='Time Series Forecast', xaxis_title='Time', yaxis_title='Value')
        return fig

    def _resolve_method(self, series: pd.Series) -> str:
        """Auto-select best method based on series characteristics."""
        s = self._detect_seasonality(series)
        if s and s > 1:
            return "sarima" if len(series) > 2 * s else "exp_smoothing"
        return "arima"

    @staticmethod
    def _detect_seasonality(series: pd.Series) -> int:
        """Detect dominant seasonality using autocorrelation."""
        from statsmodels.tsa.stattools import acf
        try:
            n_lags = min(len(series) // 2, 100)
            if n_lags < 4:
                return 1
            acf_vals = acf(series.dropna(), nlags=n_lags, fft=True)
            # Find first significant peak after lag 1
            peaks = []
            for i in range(2, len(acf_vals) - 1):
                if acf_vals[i] > acf_vals[i-1] and acf_vals[i] > acf_vals[i+1] and acf_vals[i] > 0.1:
                    peaks.append((i, acf_vals[i]))
            if peaks:
                return peaks[0][0]
        except Exception:
            pass
        return 1

    @staticmethod
    def _auto_arima_order(series: pd.Series) -> Tuple[int, int, int]:
        """Simple auto-ARIMA order selection via AIC."""
        import statsmodels.api as sm
        best_aic = np.inf
        best_order = (1, 1, 1)
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = sm.tsa.ARIMA(series, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except Exception:
                        continue
        return best_order


if __name__ == '__main__':
    # Sample data
    np.random.seed(42)
    t = np.arange(200)
    series = pd.Series(50 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(200) * 3)

    logger.info("=" * 60)
    logger.info("TIME SERIES FORECASTING")
    logger.info("=" * 60)

    # ARIMA
    forecaster = TimeSeriesForecaster(method='arima', horizon=20)
    forecaster.fit(series[:180])
    forecast = forecaster.predict()
    logger.info(f"\n🎯 Forecast (next 20): mean={forecast.mean():.2f}")

    # Walk-forward CV
    logger.info("\n--- Walk-forward CV ---")
    cv_results = forecaster.walk_forward_cv(series, n_splits=5, horizon=10)

    # Exponential Smoothing
    logger.info("\n--- Exponential Smoothing ---")
    ets = TimeSeriesForecaster(method='exp_smoothing', horizon=20, seasonal_period=12)
    ets.fit(series[:180])
    fc_ets = ets.predict()
    logger.info(f"🎯 ETS Forecast mean: {fc_ets.mean():.2f}")
