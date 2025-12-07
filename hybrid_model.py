"""
Hybrid SARIMA-LSTM Model
Combines SARIMA for linear/seasonal patterns and LSTM for nonlinear residuals
"""
import pandas as pd
import numpy as np
from sarima_model import SARIMAModel
from lstm_model import LSTMModel
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class HybridSARIMALSTM:
    """Hybrid model combining SARIMA and LSTM"""
    
    def __init__(self, sarima_order: tuple = (1, 1, 1),
                 sarima_seasonal_order: tuple = (1, 1, 1, 24),
                 lstm_sequence_length: int = 30,
                 lstm_units: tuple = (64, 32)):
        """
        Initialize hybrid model
        
        Parameters:
        -----------
        sarima_order : tuple - SARIMA (p, d, q) order
        sarima_seasonal_order : tuple - SARIMA seasonal (P, D, Q, s) order
        lstm_sequence_length : int - LSTM lookback window
        lstm_units : tuple - LSTM layer sizes
        """
        self.sarima = SARIMAModel(order=sarima_order, seasonal_order=sarima_seasonal_order)
        self.lstm = LSTMModel(sequence_length=lstm_sequence_length, lstm_units=lstm_units)
        self.is_fitted = False
        self.feature_columns = None
        self.training_data = None
    
    def fit(self, data: pd.Series, features: Optional[pd.DataFrame] = None,
            lstm_epochs: int = 50, lstm_batch_size: int = 32):
        """
        Fit the hybrid model
        
        Parameters:
        -----------
        data : pd.Series - Target time series (e.g., temperature)
        features : pd.DataFrame - Additional features (dewpoint, pressure, etc.)
        lstm_epochs : int - LSTM training epochs
        lstm_batch_size : int - LSTM batch size
        """
        print("Fitting SARIMA model...")
        # Step 1: Fit SARIMA
        self.sarima.fit(data, auto_select=True)
        
        if not self.sarima.is_fitted:
            print("SARIMA fitting failed")
            return
        
        # Step 2: Get SARIMA predictions and residuals
        print("Computing residuals...")
        fitted_values = self.sarima.fitted_model.fittedvalues
        residuals = data.loc[fitted_values.index] - fitted_values
        
        # Store feature columns if provided
        if features is not None:
            self.feature_columns = features.columns.tolist()
            # Align features with residuals
            common_idx = residuals.index.intersection(features.index)
            if len(common_idx) > 0:
                features_aligned = features.loc[common_idx]
            else:
                features_aligned = None
        else:
            features_aligned = None
        
        # Step 3: Fit LSTM on residuals
        print("Fitting LSTM model on residuals...")
        self.lstm.fit(residuals, features_aligned, 
                     epochs=lstm_epochs, batch_size=lstm_batch_size)
        
        self.is_fitted = self.sarima.is_fitted and self.lstm.is_fitted
        self.training_data = data
        
        if self.is_fitted:
            print("Hybrid model fitted successfully!")
        else:
            print("Hybrid model fitting completed with warnings")
    
    def predict(self, steps: int, start_date: Optional[pd.Timestamp] = None,
                future_features: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate hybrid predictions
        
        Parameters:
        -----------
        steps : int - Number of steps to predict
        start_date : pd.Timestamp - Start date for predictions
        future_features : pd.DataFrame - Future feature values
        
        Returns:
        --------
        pd.Series - Combined predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Step 1: Get SARIMA predictions
        sarima_pred = self.sarima.predict(steps, start_date)
        
        # Step 2: Get last residuals for LSTM
        residuals = self.sarima.get_residuals()
        if len(residuals) < self.lstm.sequence_length:
            # If not enough residuals, use zeros or simple forecast
            lstm_residual_pred = pd.Series([0] * steps, index=sarima_pred.index)
        else:
            # Get last sequence_length residuals
            last_residuals = residuals[-self.lstm.sequence_length:]
            
            # Prepare future features for LSTM if provided
            future_feat_df = None
            if future_features is not None and self.feature_columns:
                future_feat_df = future_features[self.feature_columns]
            
            # Step 3: Predict LSTM residuals
            try:
                lstm_residual_pred = self.lstm.predict_residuals(
                    last_residuals, steps, future_feat_df
                )
                lstm_residual_pred.index = sarima_pred.index
            except Exception as e:
                print(f"Error in LSTM prediction: {e}")
                lstm_residual_pred = pd.Series([0] * steps, index=sarima_pred.index)
        
        # Step 4: Combine predictions
        hybrid_pred = sarima_pred + lstm_residual_pred
        
        return hybrid_pred
    
    def evaluate(self, test_data: pd.Series, 
                 test_features: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Parameters:
        -----------
        test_data : pd.Series - Test time series
        test_features : pd.DataFrame - Test features
        
        Returns:
        --------
        Dict with MAE, RMSE, MAPE metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        steps = len(test_data)
        start_date = test_data.index[0]
        
        predictions = self.predict(steps, start_date, test_features)
        
        # Align predictions with test data
        common_idx = predictions.index.intersection(test_data.index)
        pred_aligned = predictions.loc[common_idx]
        test_aligned = test_data.loc[common_idx]
        
        # Calculate metrics (with protection against division by zero)
        mae = np.mean(np.abs(pred_aligned - test_aligned))
        rmse = np.sqrt(np.mean((pred_aligned - test_aligned) ** 2))
        mape = np.mean(np.abs((pred_aligned - test_aligned) / (test_aligned + 1e-8))) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
