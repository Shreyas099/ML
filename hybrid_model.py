"""
Hybrid SARIMA-LSTM Model (Clean Implementation)
Combines statistical and deep learning approaches for superior forecasting
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from sarima_model import SARIMAModel
from lstm_model import LSTMModel

logger = logging.getLogger(__name__)


class HybridSARIMALSTM:
    """
    Hybrid model combining SARIMA and LSTM

    Two-stage approach:
    1. SARIMA captures linear trends and seasonal patterns
    2. LSTM captures non-linear patterns in SARIMA residuals
    3. Final forecast = SARIMA forecast + LSTM residual forecast
    """

    def __init__(self,
                 sarima_order: Tuple[int, int, int] = (1, 1, 1),
                 sarima_seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24),
                 lstm_sequence_length: int = 24,
                 lstm_units: Tuple[int, int] = (32, 16)):
        """
        Initialize hybrid model

        Parameters:
        -----------
        sarima_order : tuple - ARIMA order (p,d,q)
        sarima_seasonal_order : tuple - Seasonal order (P,D,Q,s)
        lstm_sequence_length : int - LSTM lookback window
        lstm_units : tuple - LSTM layer sizes
        """
        self.sarima = SARIMAModel(
            order=sarima_order,
            seasonal_order=sarima_seasonal_order
        )
        self.lstm = LSTMModel(
            sequence_length=lstm_sequence_length,
            lstm_units=lstm_units
        )
        self.is_fitted = False

    def fit(self, data: pd.Series, features: Optional[pd.DataFrame] = None,
            lstm_epochs: int = 30, lstm_batch_size: int = 32):
        """
        Fit hybrid model

        Parameters:
        -----------
        data : pd.Series - Time series data with datetime index
        features : pd.DataFrame - Additional features (optional, not used in simple version)
        lstm_epochs : int - LSTM training epochs
        lstm_batch_size : int - LSTM batch size
        """
        if not isinstance(data, pd.Series):
            raise TypeError("data must be a pandas Series")

        if len(data) < 100:
            raise ValueError(f"Insufficient data: need at least 100 points, got {len(data)}")

        logger.info(f"Training hybrid model on {len(data)} data points")

        # Stage 1: Fit SARIMA to capture linear/seasonal patterns
        logger.info("Stage 1: Fitting SARIMA model...")
        self.sarima.fit(data)

        # Get SARIMA residuals
        residuals = self.sarima.get_residuals()
        logger.info(f"SARIMA residuals computed: {len(residuals)} points")

        # Stage 2: Fit LSTM to residuals to capture non-linear patterns
        logger.info("Stage 2: Fitting LSTM to residuals...")
        try:
            self.lstm.fit(
                residuals,
                features=None,  # Simplified - not using features
                epochs=lstm_epochs,
                batch_size=lstm_batch_size
            )
        except Exception as e:
            logger.warning(f"LSTM training failed: {e}. Using SARIMA only.")
            # If LSTM fails, we can still use SARIMA predictions
            self.lstm.is_fitted = False

        self.is_fitted = True
        logger.info("Hybrid model training complete")

    def predict(self, steps: int, start_date: pd.Timestamp,
                future_features: Optional[pd.DataFrame] = None,
                return_confidence: bool = False):
        """
        Generate forecast

        Parameters:
        -----------
        steps : int - Number of time steps to forecast
        start_date : pd.Timestamp - Start datetime for forecast
        future_features : pd.DataFrame - Future features (optional, not used)
        return_confidence : bool - Whether to return confidence intervals

        Returns:
        --------
        If return_confidence=False: forecast (pd.Series)
        If return_confidence=True: (forecast, lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        logger.info(f"Generating {steps}-step forecast")

        # Get SARIMA forecast with confidence intervals
        if return_confidence:
            sarima_pred, sarima_lower, sarima_upper = self.sarima.predict(
                steps, start_date, return_confidence=True
            )
        else:
            sarima_pred = self.sarima.predict(steps, start_date, return_confidence=False)

        # If LSTM is fitted, add residual predictions
        if self.lstm.is_fitted:
            try:
                # Get residuals for prediction
                residuals = self.sarima.get_residuals()

                if return_confidence:
                    # Get LSTM predictions with uncertainty
                    lstm_pred, lstm_std = self.lstm.predict_residuals(
                        residuals,
                        steps,
                        return_std=True
                    )

                    # Combine SARIMA + LSTM
                    hybrid_pred = sarima_pred + lstm_pred.values
                    hybrid_pred.index = sarima_pred.index

                    # Combine uncertainties (SARIMA CI + LSTM std)
                    sarima_std = (sarima_upper - sarima_lower) / (2 * 1.96)  # Convert CI to std
                    combined_std = np.sqrt(sarima_std**2 + lstm_std**2)

                    # Create combined confidence intervals
                    hybrid_lower = hybrid_pred - 1.96 * combined_std
                    hybrid_upper = hybrid_pred + 1.96 * combined_std

                    logger.info("Hybrid forecast with confidence intervals generated")
                    return hybrid_pred, hybrid_lower, hybrid_upper
                else:
                    # Simple prediction without confidence
                    lstm_pred = self.lstm.predict_residuals(residuals, steps, return_std=False)
                    hybrid_pred = sarima_pred + lstm_pred.values
                    hybrid_pred.index = sarima_pred.index

                    logger.info("Hybrid forecast generated")
                    return hybrid_pred

            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}. Using SARIMA only.")
                # Fall back to SARIMA-only predictions
                if return_confidence:
                    return sarima_pred, sarima_lower, sarima_upper
                else:
                    return sarima_pred
        else:
            # LSTM not fitted, use SARIMA only
            logger.info("LSTM not available, using SARIMA predictions only")
            if return_confidence:
                return sarima_pred, sarima_lower, sarima_upper
            else:
                return sarima_pred

    def get_component_predictions(self, steps: int, start_date: pd.Timestamp):
        """
        Get individual model predictions for comparison

        Parameters:
        -----------
        steps : int - Number of steps to forecast
        start_date : pd.Timestamp - Start datetime

        Returns:
        --------
        dict with keys:
            'sarima': SARIMA predictions
            'lstm_residuals': LSTM residual predictions (if available)
            'hybrid': Combined predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # SARIMA predictions
        sarima_pred = self.sarima.predict(steps, start_date, return_confidence=False)

        result = {'sarima': sarima_pred}

        # LSTM residual predictions if available
        if self.lstm.is_fitted:
            try:
                residuals = self.sarima.get_residuals()
                lstm_pred = self.lstm.predict_residuals(residuals, steps, return_std=False)
                result['lstm_residuals'] = lstm_pred

                # Hybrid = SARIMA + LSTM residuals
                hybrid_pred = sarima_pred + lstm_pred.values
                hybrid_pred.index = sarima_pred.index
                result['hybrid'] = hybrid_pred
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")
                result['hybrid'] = sarima_pred
        else:
            result['hybrid'] = sarima_pred

        return result

    def save(self, filepath_base: str):
        """
        Save both models

        Parameters:
        -----------
        filepath_base : str - Base path (without extension)
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        self.sarima.save(f"{filepath_base}_sarima")
        if self.lstm.is_fitted:
            self.lstm.save(f"{filepath_base}_lstm")

        logger.info(f"Hybrid model saved to {filepath_base}_*")

    @classmethod
    def load(cls, filepath_base: str):
        """
        Load both models

        Parameters:
        -----------
        filepath_base : str - Base path (without extension)

        Returns:
        --------
        HybridSARIMALSTM - Loaded model instance
        """
        instance = cls()
        instance.sarima = SARIMAModel.load(f"{filepath_base}_sarima")

        try:
            instance.lstm = LSTMModel.load(f"{filepath_base}_lstm")
        except FileNotFoundError:
            logger.warning("LSTM model not found, using SARIMA only")
            instance.lstm.is_fitted = False

        instance.is_fitted = True
        logger.info(f"Hybrid model loaded from {filepath_base}_*")
        return instance
