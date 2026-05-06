# -*- coding: utf-8 -*-
"""

████████╗██╗███╗   ███╗███████╗    ███████╗███████╗██████╗ ██╗███████╗███████╗
╚══██╔══╝██║████╗ ████║██╔════╝    ██╔════╝██╔════╝██╔══██╗██║██╔════╝██╔════╝
   ██║   ██║██╔████╔██║█████╗      ███████╗█████╗  ██████╔╝██║█████╗  ███████╗
   ██║   ██║██║╚██╔╝██║██╔══╝      ╚════██║██╔══╝  ██╔══██╗██║██╔══╝  ╚════██║
   ██║   ██║██║ ╚═╝ ██║███████╗    ███████║███████╗██║  ██║██║███████╗███████║
   ╚═╝   ╚═╝╚═╝     ╚═╝╚══════╝    ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝

 █████╗ ███╗   ██╗ ██████╗ ███╗   ███╗ █████╗ ██╗  ██╗   ██╗
██╔══██╗████╗  ██║██╔═══██╗████╗ ████║██╔══██╗██║  ╚██╗ ██╔╝
███████║██╔██╗ ██║██║   ██║██╔████╔██║███████║██║   ╚████╔╝
██╔══██║██║╚██╗██║██║   ██║██║╚██╔╝██║██╔══██║██║    ╚██╔╝
██║  ██║██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██║  ██║███████╗██║
╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝

Time Series Anomaly Detection module for scomp_link.

Methods:
  1. Conv1D Autoencoder — learns normal temporal patterns, flags high reconstruction error
  2. Moving Average / Moving Median — flags deviations beyond N standard deviations
  3. ARIMA Residuals — flags large residuals from fitted ARIMA model

"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple


class TimeSeriesAnomalyDetector:
    """
    Multi-method anomaly detection for univariate time series.

    Parameters
    ----------
    methods : list of str, optional
        Methods to use. Default: all available.
        Options: 'autoencoder', 'moving_avg', 'moving_median', 'arima'
    time_steps : int, default=288
        Sequence length for autoencoder (e.g., 288 = one day at 5-min intervals).
    ae_epochs : int, default=50
        Training epochs for autoencoder.
    ae_batch_size : int, default=128
        Batch size for autoencoder training.
    window_size : int, default=48
        Window size for moving average/median methods.
    n_sigma : float, default=3.0
        Number of standard deviations for threshold in statistical methods.
    arima_order : tuple, default=(5, 1, 0)
        ARIMA (p, d, q) order.
    threshold_percentile : float, default=95.0
        Percentile for autoencoder reconstruction error threshold.
    verbose : bool, default=True
        Print progress.

    Example
    -------
    >>> from scomp_link.models.ts_anomaly_detector import TimeSeriesAnomalyDetector
    >>> import pandas as pd
    >>>
    >>> # df_train: normal time series, df_test: time series with anomalies
    >>> detector = TimeSeriesAnomalyDetector(methods=['autoencoder', 'moving_avg'])
    >>> detector.fit(df_train['value'].values)
    >>> results = detector.detect(df_test['value'].values)
    >>> print(results['anomalies'])
    """

    AVAILABLE_METHODS = ['autoencoder', 'moving_avg', 'moving_median', 'arima']

    def __init__(
        self,
        methods: Optional[List[str]] = None,
        time_steps: int = 288,
        ae_epochs: int = 50,
        ae_batch_size: int = 128,
        window_size: int = 48,
        n_sigma: float = 3.0,
        arima_order: Tuple[int, int, int] = (5, 1, 0),
        threshold_percentile: float = 95.0,
        verbose: bool = True,
    ):
        self.methods = methods or self.AVAILABLE_METHODS
        self.time_steps = time_steps
        self.ae_epochs = ae_epochs
        self.ae_batch_size = ae_batch_size
        self.window_size = window_size
        self.n_sigma = n_sigma
        self.arima_order = arima_order
        self.threshold_percentile = threshold_percentile
        self.verbose = verbose

        self.scaler_ = None
        self.ae_model_ = None
        self.ae_threshold_ = None
        self.training_mean_ = None
        self.training_std_ = None

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    # Autoencoder
    # ------------------------------------------------------------------

    def _build_autoencoder(self, input_shape):
        """Build Conv1D autoencoder (Keras)."""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            raise ImportError(
                "TensorFlow/Keras required for autoencoder method. "
                "Install with: pip install tensorflow"
            )

        seq_len = input_shape[0]

        inputs = layers.Input(shape=input_shape)
        # Encoder
        x = layers.Conv1D(filters=32, kernel_size=7, padding="same", strides=2, activation="relu")(inputs)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1D(filters=16, kernel_size=7, padding="same", strides=2, activation="relu")(x)
        # Decoder
        x = layers.Conv1DTranspose(filters=16, kernel_size=7, padding="same", strides=2, activation="relu")(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1DTranspose(filters=32, kernel_size=7, padding="same", strides=2, activation="relu")(x)
        x = layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same")(x)
        # Crop to match input length
        x = layers.Lambda(lambda t: t[:, :seq_len, :])(x)

        model = keras.Model(inputs, x)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        return model

    def _create_sequences(self, values: np.ndarray) -> np.ndarray:
        """Create overlapping sequences for autoencoder."""
        output = []
        for i in range(len(values) - self.time_steps + 1):
            output.append(values[i: i + self.time_steps])
        return np.stack(output)

    def _fit_autoencoder(self, values: np.ndarray):
        """Train autoencoder on normal data."""
        from tensorflow import keras

        self._log("  [AE] Building and training Conv1D Autoencoder...")
        normalized = (values - self.training_mean_) / self.training_std_
        x_train = self._create_sequences(normalized.reshape(-1, 1))

        self.ae_model_ = self._build_autoencoder((x_train.shape[1], x_train.shape[2]))
        self.ae_model_.fit(
            x_train, x_train,
            epochs=self.ae_epochs,
            batch_size=self.ae_batch_size,
            validation_split=0.1,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")],
            verbose=0,
        )

        # Compute threshold from training reconstruction error
        x_pred = self.ae_model_.predict(x_train, verbose=0)
        train_mae = np.mean(np.abs(x_pred - x_train), axis=1)
        self.ae_threshold_ = np.percentile(train_mae, self.threshold_percentile)
        self._log(f"  [AE] Threshold (MAE): {self.ae_threshold_:.6f}")

    def _detect_autoencoder(self, values: np.ndarray) -> np.ndarray:
        """Detect anomalies using autoencoder reconstruction error."""
        normalized = (values - self.training_mean_) / self.training_std_
        x_test = self._create_sequences(normalized.reshape(-1, 1))
        x_pred = self.ae_model_.predict(x_test, verbose=0)
        test_mae = np.mean(np.abs(x_pred - x_test), axis=1).flatten()

        # Map back to original length (pad beginning with False)
        anomalies = np.zeros(len(values), dtype=bool)
        ae_flags = test_mae > self.ae_threshold_
        # Each sequence covers time_steps points; flag the last point of anomalous sequences
        for i in range(len(ae_flags)):
            if ae_flags[i]:
                anomalies[i + self.time_steps - 1] = True

        return anomalies

    # ------------------------------------------------------------------
    # Moving Average
    # ------------------------------------------------------------------

    def _detect_moving_avg(self, values: np.ndarray) -> np.ndarray:
        """Detect anomalies using moving average ± n_sigma."""
        series = pd.Series(values)
        rolling_mean = series.rolling(window=self.window_size, center=True).mean()
        rolling_std = series.rolling(window=self.window_size, center=True).std()

        upper = rolling_mean + self.n_sigma * rolling_std
        lower = rolling_mean - self.n_sigma * rolling_std

        anomalies = (series > upper) | (series < lower)
        return anomalies.fillna(False).values.astype(bool)

    # ------------------------------------------------------------------
    # Moving Median
    # ------------------------------------------------------------------

    def _detect_moving_median(self, values: np.ndarray) -> np.ndarray:
        """Detect anomalies using moving median ± n_sigma * MAD."""
        series = pd.Series(values)
        rolling_med = series.rolling(window=self.window_size, center=True).median()
        # MAD (Median Absolute Deviation) as robust std estimate
        rolling_mad = series.rolling(window=self.window_size, center=True).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        )
        # Scale MAD to approximate std
        rolling_mad_scaled = rolling_mad * 1.4826

        upper = rolling_med + self.n_sigma * rolling_mad_scaled
        lower = rolling_med - self.n_sigma * rolling_mad_scaled

        anomalies = (series > upper) | (series < lower)
        return anomalies.fillna(False).values.astype(bool)

    # ------------------------------------------------------------------
    # ARIMA Residuals
    # ------------------------------------------------------------------

    def _detect_arima(self, values: np.ndarray) -> np.ndarray:
        """Detect anomalies using ARIMA model residuals."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            raise ImportError(
                "statsmodels required for ARIMA method. "
                "Install with: pip install statsmodels"
            )

        self._log("  [ARIMA] Fitting model...")
        model = ARIMA(values, order=self.arima_order)
        fitted = model.fit()
        residuals = fitted.resid

        # Flag residuals beyond n_sigma standard deviations
        res_mean = np.mean(residuals)
        res_std = np.std(residuals)
        anomalies = np.abs(residuals - res_mean) > self.n_sigma * res_std

        return anomalies.astype(bool)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, values: np.ndarray):
        """
        Fit the detector on normal (non-anomalous) training data.
        Only needed for autoencoder method.

        Parameters
        ----------
        values : np.ndarray
            1D array of normal time series values.
        """
        self.training_mean_ = np.mean(values)
        self.training_std_ = np.std(values)

        if 'autoencoder' in self.methods:
            self._fit_autoencoder(values)

        return self

    def detect(self, values: np.ndarray) -> Dict:
        """
        Detect anomalies in a time series.

        Parameters
        ----------
        values : np.ndarray
            1D array of time series values to check for anomalies.

        Returns
        -------
        dict with keys:
            - 'anomalies': boolean array (True = anomaly)
            - 'methods': dict of per-method boolean arrays
            - 'consensus_score': int array (how many methods flagged each point)
        """
        n = len(values)
        method_results = {}

        method_map = {
            'autoencoder': self._detect_autoencoder,
            'moving_avg': self._detect_moving_avg,
            'moving_median': self._detect_moving_median,
            'arima': self._detect_arima,
        }

        for method in self.methods:
            if method not in method_map:
                raise ValueError(f"Unknown method '{method}'. Available: {self.AVAILABLE_METHODS}")
            self._log(f"  Running {method}...")
            result = method_map[method](values)
            method_results[method] = result

        # Consensus
        consensus = np.zeros(n, dtype=int)
        for flags in method_results.values():
            consensus += flags.astype(int)

        # Anomaly if at least 1 method flags it (configurable)
        anomalies = consensus >= 1

        self._log(f"  Total anomalies detected: {anomalies.sum()} / {n}")

        return {
            'anomalies': anomalies,
            'methods': method_results,
            'consensus_score': consensus,
        }
