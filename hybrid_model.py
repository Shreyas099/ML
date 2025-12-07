"""
Hybrid SARIMA-LSTM Model
Combines SARIMA for linear/seasonal patterns and LSTM for nonlinear residuals
"""
import pandas as pd
import numpy as np
import pickle
import os
import logging
from pathlib import Path
from sarima_model import SARIMAModel
from lstm_model import LSTMModel
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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
        # Input validation
        if not isinstance(data, pd.Series):
            raise TypeError("data must be a pandas Series")

        if len(data) < 100:
            raise ValueError(f"Insufficient data: need at least 100 points, got {len(data)}")

        if data.isna().all():
            raise ValueError("data contains only NaN values")

        if lstm_epochs < 1:
            raise ValueError(f"lstm_epochs must be positive, got {lstm_epochs}")

        if lstm_batch_size < 1:
            raise ValueError(f"lstm_batch_size must be positive, got {lstm_batch_size}")

        if features is not None:
            if not isinstance(features, pd.DataFrame):
                raise TypeError("features must be a pandas DataFrame or None")
            if len(features) != len(data):
                logger.warning(f"Feature length ({len(features)}) != data length ({len(data)}). Will align automatically.")

        logger.info("Fitting SARIMA model...")
        # Step 1: Fit SARIMA
        self.sarima.fit(data, auto_select=True)

        if not self.sarima.is_fitted:
            logger.error("SARIMA fitting failed")
            return

        # Step 2: Get SARIMA predictions and residuals
        logger.info("Computing residuals...")
        fitted_values = self.sarima.fitted_model.fittedvalues
        residuals = data.loc[fitted_values.index] - fitted_values
        
        # Store feature columns if provided
        if features is not None:
            self.feature_columns = features.columns.tolist()
            # Align features with residuals
            common_idx = residuals.index.intersection(features.index)
            if len(common_idx) > 0:
                features_aligned = features.loc[common_idx]
                # Warn if significant data loss during alignment
                alignment_ratio = len(common_idx) / len(residuals)
                if alignment_ratio < 0.8:
                    logger.warning(f"Feature alignment resulted in {alignment_ratio:.1%} data retention. " +
                                 f"Lost {len(residuals) - len(common_idx)} residual points.")
                else:
                    logger.info(f"Feature alignment successful: {len(common_idx)} points aligned")
            else:
                logger.error("No overlapping indices between features and residuals. Features will be ignored.")
                features_aligned = None
        else:
            features_aligned = None
        
        # Step 3: Fit LSTM on residuals
        logger.info("Fitting LSTM model on residuals...")
        self.lstm.fit(residuals, features_aligned,
                     epochs=lstm_epochs, batch_size=lstm_batch_size)

        self.is_fitted = self.sarima.is_fitted and self.lstm.is_fitted
        self.training_data = data

        if self.is_fitted:
            logger.info("Hybrid model fitted successfully!")
        else:
            logger.warning("Hybrid model fitting completed with warnings")
    
    def predict(self, steps: int, start_date: Optional[pd.Timestamp] = None,
                future_features: Optional[pd.DataFrame] = None,
                return_confidence: bool = False) -> pd.Series:
        """
        Generate hybrid predictions

        Parameters:
        -----------
        steps : int - Number of steps to predict
        start_date : pd.Timestamp - Start date for predictions
        future_features : pd.DataFrame - Future feature values
        return_confidence : bool - If True, return (predictions, lower_bound, upper_bound)

        Returns:
        --------
        pd.Series - Combined predictions
        or tuple: (predictions, lower_bound, upper_bound) if return_confidence=True
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Input validation
        if not isinstance(steps, int) or steps < 1:
            raise ValueError(f"steps must be a positive integer, got {steps}")

        if steps > 1000:
            logger.warning(f"Predicting {steps} steps ahead may be unreliable")

        if start_date is not None and not isinstance(start_date, pd.Timestamp):
            raise TypeError("start_date must be a pandas Timestamp or None")

        if future_features is not None:
            if not isinstance(future_features, pd.DataFrame):
                raise TypeError("future_features must be a pandas DataFrame or None")
            if len(future_features) != steps:
                raise ValueError(f"future_features length ({len(future_features)}) must match steps ({steps})")

        # Step 1: Get SARIMA predictions with confidence intervals
        sarima_result = self.sarima.predict(steps, start_date, return_confidence=return_confidence)

        if return_confidence:
            sarima_pred, sarima_lower, sarima_upper = sarima_result
        else:
            sarima_pred = sarima_result

        # Step 2: Get last residuals for LSTM
        residuals = self.sarima.get_residuals()
        if len(residuals) < self.lstm.sequence_length:
            # If not enough residuals, use zeros or simple forecast
            lstm_residual_pred = pd.Series([0] * steps, index=sarima_pred.index)
            lstm_uncertainty = 0
        else:
            # Get last sequence_length residuals
            last_residuals = residuals[-self.lstm.sequence_length:]

            # Prepare future features for LSTM if provided
            future_feat_df = None
            if future_features is not None and self.feature_columns:
                future_feat_df = future_features[self.feature_columns]

            # Step 3: Predict LSTM residuals with uncertainty
            try:
                if return_confidence and self.lstm.is_fitted:
                    lstm_residual_pred, lstm_std = self.lstm.predict_residuals(
                        last_residuals, steps, future_feat_df, return_std=True
                    )
                    lstm_residual_pred.index = sarima_pred.index
                    lstm_uncertainty = lstm_std
                else:
                    lstm_residual_pred = self.lstm.predict_residuals(
                        last_residuals, steps, future_feat_df
                    )
                    lstm_residual_pred.index = sarima_pred.index
                    lstm_uncertainty = lstm_residual_pred.std() if return_confidence else 0
            except Exception as e:
                logger.error(f"Error in LSTM prediction: {e}")
                lstm_residual_pred = pd.Series([0] * steps, index=sarima_pred.index)
                lstm_uncertainty = 0

        # Step 4: Combine predictions
        hybrid_pred = sarima_pred + lstm_residual_pred

        if return_confidence:
            # Combine uncertainties (assuming independence)
            # Add LSTM uncertainty to SARIMA intervals
            hybrid_lower = sarima_lower + lstm_residual_pred - lstm_uncertainty * 1.96
            hybrid_upper = sarima_upper + lstm_residual_pred + lstm_uncertainty * 1.96
            return hybrid_pred, hybrid_lower, hybrid_upper
        else:
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

    def save(self, filepath: str):
        """
        Save the hybrid model to disk

        Parameters:
        -----------
        filepath : str - Path to save the model (without extension)
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save SARIMA model
        self.sarima.save(f"{filepath}_sarima.pkl")

        # Save LSTM model
        self.lstm.save(f"{filepath}_lstm")

        # Save hybrid model metadata
        metadata = {
            'is_fitted': self.is_fitted,
            'feature_columns': self.feature_columns,
            'training_data': self.training_data
        }

        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'HybridSARIMALSTM':
        """
        Load a hybrid model from disk

        Parameters:
        -----------
        filepath : str - Path to load the model from (without extension)

        Returns:
        --------
        HybridSARIMALSTM - Loaded model instance
        """
        # Load metadata
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)

        # Create instance (parameters will be overridden by loaded models)
        instance = cls()

        # Load SARIMA model
        instance.sarima = SARIMAModel.load(f"{filepath}_sarima.pkl")

        # Load LSTM model
        instance.lstm = LSTMModel.load(f"{filepath}_lstm")

        # Restore metadata
        instance.is_fitted = metadata['is_fitted']
        instance.feature_columns = metadata['feature_columns']
        instance.training_data = metadata['training_data']

        logger.info(f"Model loaded from {filepath}")
        return instance
