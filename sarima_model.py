"""
SARIMA Model Implementation
Handles linear and seasonal patterns in time series data
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


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
        # Remove NaN values
        data_clean = data.dropna()
        
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
            print(f"Error fitting SARIMA model: {e}")
            # Fallback to simpler model
            try:
                self.model = SARIMAX(data_clean, 
                                   order=(1, 1, 1), 
                                   seasonal_order=(1, 1, 1, 24),
                                   enforce_stationarity=False,
                                   enforce_invertibility=False)
                self.fitted_model = self.model.fit(disp=False, maxiter=50)
                self.is_fitted = True
            except Exception as e2:
                print(f"Error with fallback model: {e2}")
                self.is_fitted = False
    
    def predict(self, steps: int, start: Optional[pd.Timestamp] = None) -> pd.Series:
        """
        Generate predictions
        
        Parameters:
        -----------
        steps : int - Number of steps ahead to predict
        start : pd.Timestamp - Start date for prediction
        
        Returns:
        --------
        pd.Series - Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            if start is not None:
                forecast.index = pd.date_range(start=start, periods=steps, freq='H')
            return forecast
        except Exception as e:
            print(f"Error in SARIMA prediction: {e}")
            # Return naive forecast as fallback
            if start is not None:
                return pd.Series([self.fitted_model.forecast(steps=1).iloc[0]] * steps,
                                index=pd.date_range(start=start, periods=steps, freq='H'))
            return pd.Series([0] * steps)
    
    def get_residuals(self) -> pd.Series:
        """Get residuals from fitted model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting residuals")
        return self.fitted_model.resid
