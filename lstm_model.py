"""
LSTM Model Implementation (Simplified and Robust)
Captures nonlinear patterns in residual time series
"""
import pandas as pd
import numpy as np
import os
import pickle
import logging

# Optimize TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class LSTMModel:
    """Simplified LSTM model for capturing nonlinear patterns in residuals"""

    def __init__(self, sequence_length: int = 24,
                 lstm_units: Tuple[int, int] = (32, 16),
                 dropout_rate: float = 0.2):
        """
        Initialize LSTM model with simple architecture

        Parameters:
        -----------
        sequence_length : int - Lookback window (default: 24 hours for daily pattern)
        lstm_units : tuple - Units in each LSTM layer (default: 32, 16 for speed)
        dropout_rate : float - Dropout for regularization
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.is_fitted = False

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training

        Parameters:
        -----------
        data : np.ndarray - 1D array of residuals

        Returns:
        --------
        X, y : Training sequences and targets
        """
        if len(data) < self.sequence_length + 1:
            raise ValueError(f"Need at least {self.sequence_length + 1} points, got {len(data)}")

        X, y = [], []

        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])

        X = np.array(X).reshape(-1, self.sequence_length, 1)
        y = np.array(y)

        return X, y

    def build_model(self):
        """Build simple, fast LSTM architecture"""
        model = Sequential([
            LSTM(self.lstm_units[0], return_sequences=True,
                 input_shape=(self.sequence_length, 1)),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units[1], return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def fit(self, residuals: pd.Series, features: Optional[pd.DataFrame] = None,
            epochs: int = 30, batch_size: int = 32, validation_split: float = 0.1):
        """
        Fit LSTM model to residuals

        Parameters:
        -----------
        residuals : pd.Series - Residuals from SARIMA model
        features : pd.DataFrame - Ignored in this simplified version
        epochs : int - Training epochs (default: 30 for speed)
        batch_size : int - Batch size
        validation_split : float - Validation split ratio
        """
        try:
            # Clean and prepare data
            residuals_clean = residuals.dropna()

            if len(residuals_clean) < self.sequence_length + 10:
                raise ValueError(
                    f"Insufficient data: need at least {self.sequence_length + 10} points, "
                    f"got {len(residuals_clean)}"
                )

            # Scale residuals
            residuals_values = residuals_clean.values.reshape(-1, 1)
            residuals_scaled = self.scaler.fit_transform(residuals_values).flatten()

            logger.info(f"Preparing LSTM sequences from {len(residuals_scaled)} residuals")

            # Create sequences
            X, y = self.create_sequences(residuals_scaled)

            if len(X) == 0:
                raise ValueError("No training sequences created")

            logger.info(f"Created {len(X)} training sequences")

            # Build model
            self.model = self.build_model()

            # Train model
            logger.info(f"Training LSTM for {epochs} epochs...")

            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=0,
                shuffle=False  # Don't shuffle time series
            )

            self.is_fitted = True
            final_loss = history.history['loss'][-1]
            logger.info(f"LSTM trained successfully (final loss: {final_loss:.4f})")

        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            self.is_fitted = False
            raise

    def predict_residuals(self, last_residuals: pd.Series, steps: int,
                         future_features: Optional[pd.DataFrame] = None,
                         return_std: bool = False):
        """
        Predict future residuals

        Parameters:
        -----------
        last_residuals : pd.Series - Last N residuals (at least sequence_length)
        steps : int - Number of steps to predict
        future_features : pd.DataFrame - Ignored in this simplified version
        return_std : bool - If True, also return standard deviation

        Returns:
        --------
        pd.Series or (pd.Series, np.ndarray) - Predicted residuals, optionally with std
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Prepare last sequence
        if len(last_residuals) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} points for prediction, "
                f"got {len(last_residuals)}"
            )

        # Get last sequence and scale
        last_values = last_residuals.values[-self.sequence_length:].reshape(-1, 1)
        last_scaled = self.scaler.transform(last_values).flatten()

        # Predict iteratively
        if return_std:
            # Monte Carlo approach: multiple predictions for uncertainty
            n_iterations = 10
            all_predictions = []

            for _ in range(n_iterations):
                preds = self._predict_sequence(last_scaled.copy(), steps)
                all_predictions.append(preds)

            all_predictions = np.array(all_predictions)
            mean_pred = all_predictions.mean(axis=0)
            std_pred = all_predictions.std(axis=0)

            return pd.Series(mean_pred), std_pred
        else:
            predictions = self._predict_sequence(last_scaled, steps)
            return pd.Series(predictions)

    def _predict_sequence(self, last_sequence: np.ndarray, steps: int) -> np.ndarray:
        """
        Internal method to predict a sequence iteratively

        Parameters:
        -----------
        last_sequence : np.ndarray - Last sequence_length values (scaled)
        steps : int - Number of steps to predict

        Returns:
        --------
        np.ndarray - Predicted values (unscaled)
        """
        predictions = []
        current_seq = last_sequence.copy()

        for _ in range(steps):
            # Prepare input
            X_pred = current_seq.reshape(1, self.sequence_length, 1)

            # Predict
            pred_scaled = self.model.predict(X_pred, verbose=0)[0, 0]
            predictions.append(pred_scaled)

            # Update sequence (shift left, add new prediction)
            current_seq = np.append(current_seq[1:], pred_scaled)

        # Inverse transform to original scale
        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_unscaled = self.scaler.inverse_transform(predictions_array).flatten()

        return predictions_unscaled

    def save(self, filepath: str):
        """Save model to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        # Save Keras model
        self.model.save(f"{filepath}.keras")

        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }

        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"LSTM model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'LSTMModel':
        """Load model from disk"""
        # Load metadata
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)

        # Create instance
        instance = cls(
            sequence_length=metadata['sequence_length'],
            lstm_units=metadata['lstm_units'],
            dropout_rate=metadata['dropout_rate']
        )

        # Load Keras model
        instance.model = load_model(f"{filepath}.keras")
        instance.scaler = metadata['scaler']
        instance.is_fitted = metadata['is_fitted']

        logger.info(f"LSTM model loaded from {filepath}")
        return instance
