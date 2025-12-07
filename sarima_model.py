"""
SARIMA Model Implementation
Handles linear and seasonal patterns in time series data
"""
import pandas as pd
import numpy as np
import pickle
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


class SARIMAModel:
    """SARIMA model for capturing linear and seasonal patterns"""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24)):
        """
        Initialize SARIMA model

        Parameters:
        -----------
        order : (p, d, q) - ARIMA order
        seasonal_order : (P, D, Q, s) - Seasonal order with period s
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        self.data_freq = None  # Store inferred frequency from training data
    
    def check_stationarity(self, series: pd.Series) -> bool:
        """Check if series is stationary using Augmented Dickey-Fuller test"""
        result = adfuller(series.dropna())
        return result[1] <= 0.05  # p-value <= 0.05 means stationary
    
    def make_stationary(self, series: pd.Series) -> Tuple[pd.Series, int]:
        """Make series stationary through differencing"""
        d = 0
        series_diff = series.copy()
        
        while not self.check_stationarity(series_diff) and d < 2:
            series_diff = series_diff.diff().dropna()
            d += 1
        
        return series_diff, d
    
    def auto_select_order(self, series: pd.Series, max_p: int = 3, max_q: int = 3,
                         max_P: int = 2, max_Q: int = 2, s: int = 24) -> Tuple[Tuple, Tuple]:
        """
        Automatically select SARIMA order using AIC
        Simplified version - in production, use auto_arima or similar
        """
        best_aic = np.inf
        best_order = (1, 1, 1)
        best_seasonal_order = (1, 1, 1, s)
        
        # Try a few common configurations
        orders_to_try = [
            ((1, 1, 1), (1, 1, 1, s)),
            ((2, 1, 2), (1, 1, 1, s)),
            ((1, 1, 1), (2, 1, 2, s)),
            ((1, 1, 0), (1, 1, 0, s)),
        ]
        
        for order, seasonal_order in orders_to_try:
            try:
                model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                              enforce_stationarity=False, enforce_invertibility=False)
                fitted = model.fit(disp=False, maxiter=50)
                aic = fitted.aic
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = order
                    best_seasonal_order = seasonal_order
            except Exception:
                continue
        
        return best_order, best_seasonal_order
    
    def fit(self, data: pd.Series, auto_select: bool = True):
        """
        Fit SARIMA model to data

        Parameters:
        -----------
        data : pd.Series - Time series data
        auto_select : bool - Whether to auto-select order
        """
        # Input validation
        if not isinstance(data, pd.Series):
            raise TypeError("data must be a pandas Series")

        if len(data) < 10:
            raise ValueError(f"Insufficient data: need at least 10 points, got {len(data)}")

        # Remove NaN values
        data_clean = data.dropna()

        if len(data_clean) == 0:
            raise ValueError("data contains only NaN values after cleaning")

        # Infer frequency from the data index
        if isinstance(data_clean.index, pd.DatetimeIndex):
            self.data_freq = pd.infer_freq(data_clean.index)
            if self.data_freq is None:
                # Try to infer from first few intervals
                if len(data_clean) > 1:
                    avg_delta = (data_clean.index[-1] - data_clean.index[0]) / (len(data_clean) - 1)
                    # Map common intervals to pandas freq strings
                    if avg_delta <= pd.Timedelta(hours=1, minutes=30):
                        self.data_freq = 'H'  # Hourly
                    elif avg_delta <= pd.Timedelta(days=1, hours=12):
                        self.data_freq = 'D'  # Daily
                    elif avg_delta <= pd.Timedelta(days=7, hours=12):
                        self.data_freq = 'W'  # Weekly
                    else:
                        self.data_freq = 'M'  # Monthly
                else:
                    self.data_freq = 'H'  # Default to hourly
        else:
            self.data_freq = None  # No datetime index
        
        if len(data_clean) < 100:
            # Use default orders for small datasets
            order = self.order
            seasonal_order = self.seasonal_order
        elif auto_select:
            order, seasonal_order = self.auto_select_order(data_clean)
        else:
            order = self.order
            seasonal_order = self.seasonal_order
        
        try:
            self.model = SARIMAX(data_clean, 
                               order=order, 
                               seasonal_order=seasonal_order,
                               enforce_stationarity=False,
                               enforce_invertibility=False)
            self.fitted_model = self.model.fit(disp=False, maxiter=100)
            self.is_fitted = True
            self.order = order
            self.seasonal_order = seasonal_order
        except Exception as e:
            logger.error(f"Error fitting SARIMA model: {e}")
            # Fallback to simpler model
            try:
                self.model = SARIMAX(data_clean,
                                   order=(1, 1, 1),
                                   seasonal_order=(1, 1, 1, 24),
                                   enforce_stationarity=False,
                                   enforce_invertibility=False)
                self.fitted_model = self.model.fit(disp=False, maxiter=50)
                self.is_fitted = True
                logger.info("Using fallback SARIMA model with simplified parameters")
            except Exception as e2:
                logger.error(f"Error with fallback model: {e2}")
                self.is_fitted = False
    
    def predict(self, steps: int, start: Optional[pd.Timestamp] = None,
                return_confidence: bool = False):
        """
        Generate predictions

        Parameters:
        -----------
        steps : int - Number of steps ahead to predict
        start : pd.Timestamp - Start date for prediction
        return_confidence : bool - If True, return (forecast, lower, upper)

        Returns:
        --------
        pd.Series or tuple - Predictions, or (predictions, lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        try:
            if return_confidence:
                # Get forecast with confidence intervals
                forecast_result = self.fitted_model.get_forecast(steps=steps)
                forecast = forecast_result.predicted_mean
                conf_int = forecast_result.conf_int(alpha=0.05)  # 95% confidence

                if start is not None and self.data_freq is not None:
                    idx = pd.date_range(start=start, periods=steps, freq=self.data_freq)
                    forecast.index = idx
                    conf_int.index = idx

                lower = conf_int.iloc[:, 0]
                upper = conf_int.iloc[:, 1]

                return forecast, lower, upper
            else:
                forecast = self.fitted_model.forecast(steps=steps)
                if start is not None and self.data_freq is not None:
                    forecast.index = pd.date_range(start=start, periods=steps, freq=self.data_freq)
                return forecast
        except Exception as e:
            logger.error(f"Error in SARIMA prediction: {e}")
            # Return naive forecast as fallback
            if start is not None and self.data_freq is not None:
                idx = pd.date_range(start=start, periods=steps, freq=self.data_freq)
                naive_val = self.fitted_model.forecast(steps=1).iloc[0] if hasattr(self.fitted_model, 'forecast') else 0
                forecast = pd.Series([naive_val] * steps, index=idx)

                if return_confidence:
                    # Use simple std for uncertainty
                    std = self.fitted_model.resid.std() if hasattr(self.fitted_model, 'resid') else 2.0
                    lower = forecast - 1.96 * std
                    upper = forecast + 1.96 * std
                    return forecast, lower, upper
                return forecast
            return pd.Series([0] * steps)
    
    def get_residuals(self) -> pd.Series:
        """Get residuals from fitted model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting residuals")
        return self.fitted_model.resid

    def save(self, filepath: str):
        """
        Save the SARIMA model to disk

        Parameters:
        -----------
        filepath : str - Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_data = {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'fitted_model': self.fitted_model,
            'is_fitted': self.is_fitted,
            'data_freq': self.data_freq
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, filepath: str) -> 'SARIMAModel':
        """
        Load a SARIMA model from disk

        Parameters:
        -----------
        filepath : str - Path to load the model from

        Returns:
        --------
        SARIMAModel - Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        instance = cls(
            order=model_data['order'],
            seasonal_order=model_data['seasonal_order']
        )
        instance.fitted_model = model_data['fitted_model']
        instance.is_fitted = model_data['is_fitted']
        instance.data_freq = model_data.get('data_freq', 'H')  # Default to hourly for backwards compatibility
        instance.model = None  # Will be reconstructed if needed

        return instance
