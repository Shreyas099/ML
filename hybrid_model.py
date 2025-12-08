"""
Hybrid SARIMA-LSTM Model (Improvements Applied)
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from sarima_model import SARIMAModel
from lstm_model import LSTMModel

logger = logging.getLogger(__name__)

class HybridSARIMALSTM:
    def __init__(self,
                 sarima_order: Tuple[int, int, int] = (1, 1, 1),
                 sarima_seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24),
                 lstm_sequence_length: int = 24,
                 lstm_units: Tuple[int, int] = (32, 16)):
        self.sarima = SARIMAModel(order=sarima_order, seasonal_order=sarima_seasonal_order)
        self.lstm = LSTMModel(sequence_length=lstm_sequence_length, lstm_units=lstm_units)
        self.is_fitted = False

    def fit(self, data: pd.Series, features: Optional[pd.DataFrame] = None,
            lstm_epochs: int = 30, lstm_batch_size: int = 32):
        
        if not isinstance(data, pd.Series):
            raise TypeError("data must be a pandas Series")

        logger.info(f"Training hybrid model on {len(data)} data points")

        # Stage 1: SARIMA
        logger.info("Stage 1: Fitting SARIMA model...")
        self.sarima.fit(data)
        residuals = self.sarima.get_residuals()

        # Stage 2: LSTM with Features
        logger.info("Stage 2: Fitting LSTM to residuals...")
        try:
            # Pass features explicitly here
            self.lstm.fit(
                residuals,
                features=features,  # ENABLED MULTIVARIATE SUPPORT
                epochs=lstm_epochs,
                batch_size=lstm_batch_size
            )
        except Exception as e:
            logger.warning(f"LSTM training failed: {e}. Using SARIMA only.")
            self.lstm.is_fitted = False

        self.is_fitted = True

    def predict(self, steps: int, start_date: pd.Timestamp,
                future_features: Optional[pd.DataFrame] = None,
                return_confidence: bool = False):
        
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        # SARIMA Forecast
        sarima_res = self.sarima.predict(steps, start_date, return_confidence=True)
        # Unpack tuple safely
        sarima_pred, sarima_lower, sarima_upper = sarima_res
        
        # LSTM Residual Forecast
        if self.lstm.is_fitted:
            try:
                residuals = self.sarima.get_residuals()
                
                # Get LSTM prediction (with Monte Carlo uncertainty if requested)
                lstm_res = self.lstm.predict_residuals(
                    residuals, 
                    steps, 
                    future_features=future_features, 
                    return_std=return_confidence
                )
                
                if return_confidence:
                    lstm_pred, lstm_std = lstm_res
                else:
                    lstm_pred, lstm_std = lstm_res, 0

                # Combine
                hybrid_pred = sarima_pred + lstm_pred.values
                hybrid_pred.index = sarima_pred.index

                if return_confidence:
                    # Combine uncertainties (RMSE propagation)
                    sarima_std = (sarima_upper - sarima_lower) / 3.92 # Approx 95% CI to std
                    total_std = np.sqrt(sarima_std**2 + lstm_std**2)
                    
                    hybrid_lower = hybrid_pred - 1.96 * total_std
                    hybrid_upper = hybrid_pred + 1.96 * total_std
                    return hybrid_pred, hybrid_lower, hybrid_upper
                
                return hybrid_pred

            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}. Fallback to SARIMA.")
                return (sarima_pred, sarima_lower, sarima_upper) if return_confidence else sarima_pred
        
        return (sarima_pred, sarima_lower, sarima_upper) if return_confidence else sarima_pred

    def save(self, filepath_base: str):
        self.sarima.save(f"{filepath_base}_sarima")
        if self.lstm.is_fitted:
            self.lstm.save(f"{filepath_base}_lstm")

    @classmethod
    def load(cls, filepath_base: str):
        instance = cls()
        instance.sarima = SARIMAModel.load(f"{filepath_base}_sarima")
        try:
            instance.lstm = LSTMModel.load(f"{filepath_base}_lstm")
        except:
            instance.lstm.is_fitted = False
        instance.is_fitted = True
        return instance
