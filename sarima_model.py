"""
SARIMA Model Implementation (Optimized for Speed)
Captures linear trends and seasonal patterns in time series data
"""
import pandas as pd
import numpy as np
import pickle
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SARIMAModel:
    """Fast SARIMA model with fixed parameters - optimized for web applications"""

    def __init__(self, order: Tuple[int, int, int] = (1, 0, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 24)):
        """
        Initialize SARIMA model with simple, fast parameters

        Default parameters are optimized for:
        - Fast training (< 30 seconds)
        - Hourly weather data with daily seasonality (period=24)
        - Good balance of accuracy and speed

        Parameters:
        -----------
        order : (p, d, q) - ARIMA order (default: 1,0,1 - simple AR+MA)
        seasonal_order : (P, D, Q, s) - Seasonal order (default: 1,0,1,24 - daily pattern)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model = None
        self.is_fitted = False
        self.data_freq = None

    def fit(self, data: pd.Series, max_iter: int = 100):
        """
        Fit SARIMA model to data

        Parameters:
        -----------
        data : pd.Series - Time series data with datetime index
        max_iter : int - Maximum iterations for optimization (default: 100)
        """
        if not isinstance(data, pd.Series):
            raise TypeError("data must be a pandas Series")

        if len(data) < 50:
            raise ValueError(f"Insufficient data: need at least 50 points, got {len(data)}")

        # Clean data
        data_clean = data.dropna()
        logger.info(f"Training SARIMA{self.order}x{self.seasonal_order} on {len(data_clean)} points")

        # Infer frequency
        if isinstance(data_clean.index, pd.DatetimeIndex):
            self.data_freq = pd.infer_freq(data_clean.index)
            if self.data_freq is None:
                # Estimate from average time delta
                avg_delta = (data_clean.index[-1] - data_clean.index[0]) / (len(data_clean) - 1)
                if avg_delta <= pd.Timedelta(hours=1, minutes=30):
                    self.data_freq = 'H'
                elif avg_delta <= pd.Timedelta(days=1, hours=12):
                    self.data_freq = 'D'
                else:
                    self.data_freq = 'W'

        try:
            # Fit SARIMAX model with optimized settings for speed
            model = SARIMAX(
                data_clean,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,  # Faster
                enforce_invertibility=False,  # Faster
                concentrate_scale=True  # Faster optimization
            )

            self.fitted_model = model.fit(
                disp=False,
                maxiter=max_iter,
                method='lbfgs',  # Fast optimizer
                optim_score='approx'  # Faster scoring
            )

            self.is_fitted = True
            logger.info(f"SARIMA fitted successfully (AIC: {self.fitted_model.aic:.2f})")

        except Exception as e:
            logger.error(f"Error fitting SARIMA: {e}")
            raise

    def predict(self, steps: int, start: Optional[pd.Timestamp] = None,
                return_confidence: bool = False) -> Tuple:
        """
        Generate forecast

        Parameters:
        -----------
        steps : int - Number of time steps to forecast
        start : pd.Timestamp - Start datetime for forecast (optional)
        return_confidence : bool - Whether to return confidence intervals

        Returns:
        --------
        If return_confidence=True: (forecast, lower_bound, upper_bound)
        If return_confidence=False: forecast only
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        try:
            if return_confidence:
                # Get forecast with 95% confidence intervals
                forecast_result = self.fitted_model.get_forecast(steps=steps)
                forecast = forecast_result.predicted_mean
                conf_int = forecast_result.conf_int(alpha=0.05)

                # Create proper index
                if start is not None and self.data_freq is not None:
                    idx = pd.date_range(start=start, periods=steps, freq=self.data_freq)
                    forecast.index = idx
                    conf_int.index = idx

                lower = conf_int.iloc[:, 0]
                upper = conf_int.iloc[:, 1]

                return forecast, lower, upper
            else:
                # Simple forecast without confidence intervals
                forecast = self.fitted_model.forecast(steps=steps)

                if start is not None and self.data_freq is not None:
                    idx = pd.date_range(start=start, periods=steps, freq=self.data_freq)
                    forecast.index = idx

                return forecast

        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            raise

    def get_residuals(self) -> pd.Series:
        """Get model residuals (actual - fitted)"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.fitted_model.resid

    def save(self, filepath: str):
        """Save model to file"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump({
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'fitted_model': self.fitted_model,
                'data_freq': self.data_freq
            }, f)
        logger.info(f"SARIMA model saved to {filepath}.pkl")

    @classmethod
    def load(cls, filepath: str):
        """Load model from file"""
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)

        instance = cls(order=data['order'], seasonal_order=data['seasonal_order'])
        instance.fitted_model = data['fitted_model']
        instance.data_freq = data['data_freq']
        instance.is_fitted = True

        logger.info(f"SARIMA model loaded from {filepath}.pkl")
        return instance
