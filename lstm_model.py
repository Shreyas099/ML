"""
LSTM Model Implementation
Handles nonlinear patterns in residual time series
"""
import pandas as pd
import numpy as np
import os
import pickle
import logging
# Optimize TensorFlow for CPU and reduce startup time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for faster startup

from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


class LSTMModel:
    """LSTM model for capturing nonlinear patterns in residuals"""

    def __init__(self, sequence_length: int = 30,
                 lstm_units: Tuple[int, int] = (64, 32),
                 dropout_rate: float = 0.2):
        """
        Initialize LSTM model

        Parameters:
        -----------
        sequence_length : int - Number of time steps to look back
        lstm_units : Tuple[int, int] - Number of units in each LSTM layer
        dropout_rate : float - Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = None  # Store feature scaler for prediction
        self.is_fitted = False
        self.last_training_features = None  # Store last features from training
    
    def create_sequences(self, data: np.ndarray, features: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Parameters:
        -----------
        data : np.ndarray - Target time series (residuals)
        features : np.ndarray - Additional features (optional)
        
        Returns:
        --------
        X, y : Training sequences and targets
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            if features is not None:
                # Combine residual sequence with features
                seq = np.concatenate([
                    data[i-self.sequence_length:i].reshape(-1, 1),
                    features[i-self.sequence_length:i]
                ], axis=1)
            else:
                seq = data[i-self.sequence_length:i].reshape(-1, 1)
            
            X.append(seq)
            y.append(data[i])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(self.lstm_units[0], return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units[1], return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def fit(self, residuals: pd.Series, features: Optional[pd.DataFrame] = None,
            epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2):
        """
        Fit LSTM model to residuals
        
        Parameters:
        -----------
        residuals : pd.Series - Residuals from SARIMA model
        features : pd.DataFrame - Additional features (dewpoint, pressure, etc.)
        epochs : int - Training epochs
        batch_size : int - Batch size
        validation_split : float - Validation split ratio
        """
        # Prepare data
        residuals_clean = residuals.dropna().values.reshape(-1, 1)
        
        # Scale residuals
        residuals_scaled = self.scaler.fit_transform(residuals_clean).flatten()
        
        # Prepare features if provided
        features_array = None
        if features is not None:
            features_clean = features.loc[residuals.index].dropna()
            # Align indices
            common_idx = residuals.index.intersection(features_clean.index)
            if len(common_idx) > self.sequence_length:
                residuals_aligned = residuals.loc[common_idx]
                features_aligned = features_clean.loc[common_idx]
                
                # Scale features
                from sklearn.preprocessing import StandardScaler
                self.feature_scaler = StandardScaler()
                features_scaled = self.feature_scaler.fit_transform(features_aligned.values)
                
                residuals_scaled = self.scaler.fit_transform(
                    residuals_aligned.values.reshape(-1, 1)
                ).flatten()
                
                features_array = features_scaled
        
        if len(residuals_scaled) < self.sequence_length + 10:
            logger.warning("Insufficient data for LSTM training")
            self.is_fitted = False
            return

        # Create sequences
        X, y = self.create_sequences(residuals_scaled, features_array)

        if len(X) == 0:
            logger.warning("No sequences created")
            self.is_fitted = False
            return
        
        # Build model
        if features_array is not None:
            input_shape = (self.sequence_length, 1 + features_array.shape[1])
        else:
            input_shape = (self.sequence_length, 1)
        
        self.model = self.build_model(input_shape)
        
        # Train model
        try:
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=0,
                shuffle=False
            )
            self.is_fitted = True

            # Store last sequence of features for prediction initialization
            if features_array is not None:
                self.last_training_features = features_array[-self.sequence_length:].copy()
            else:
                self.last_training_features = None
            logger.info("LSTM model trained successfully")
        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            self.is_fitted = False
    
    def predict(self, last_sequence: np.ndarray, steps: int,
                future_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate predictions
        
        Parameters:
        -----------
        last_sequence : np.ndarray - Last sequence of residuals
        steps : int - Number of steps to predict
        future_features : np.ndarray - Future feature values (optional)
        
        Returns:
        --------
        np.ndarray - Predicted residuals (scaled)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for step in range(steps):
            # Reshape for prediction
            if len(current_sequence.shape) == 2:
                X_pred = current_sequence.reshape(1, self.sequence_length, current_sequence.shape[1])
            else:
                X_pred = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Predict next value
            pred = self.model.predict(X_pred, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Update sequence
            if future_features is not None and step < len(future_features):
                if len(current_sequence.shape) == 2:
                    new_point = np.concatenate([
                        np.array([[pred]]),
                        future_features[step:step+1]
                    ], axis=1)
                else:
                    new_point = np.array([[pred]])
            else:
                new_point = np.array([[pred]])
            
            # Shift sequence
            current_sequence = np.vstack([current_sequence[1:], new_point])
        
        return np.array(predictions)
    
    def predict_residuals(self, last_residuals: pd.Series, steps: int,
                         future_features: Optional[pd.DataFrame] = None,
                         return_std: bool = False):
        """
        Predict future residuals and inverse transform

        Parameters:
        -----------
        last_residuals : pd.Series - Last N residuals (at least sequence_length)
        steps : int - Steps to predict
        future_features : pd.DataFrame - Future features
        return_std : bool - If True, return (predictions, std_dev)

        Returns:
        --------
        pd.Series or tuple - Predicted residuals (original scale), or (predictions, std)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare last sequence
        residuals_scaled = self.scaler.transform(
            last_residuals.values.reshape(-1, 1)
        ).flatten()
        
        last_seq = residuals_scaled[-self.sequence_length:]
        
        # Check if model was trained with features
        # Get expected input shape from model
        expected_features = self.model.input_shape[-1] - 1  # Subtract 1 for the residual column
        
        # Prepare future features if model expects them
        future_feat_array = None
        if expected_features > 0:
            if future_features is not None and self.feature_scaler is not None:
                # Use the same scaler from training
                future_feat_array = self.feature_scaler.transform(future_features.values)
            else:
                # Model expects features but none provided - create dummy zeros
                # This is a fallback; predictions won't be as accurate
                future_feat_array = np.zeros((steps, expected_features))

            # Use actual last training features if available, otherwise use zeros
            if self.last_training_features is not None:
                # Use the actual features from the end of training
                initial_features = self.last_training_features.copy()
            else:
                # Fallback to zeros if features weren't stored
                initial_features = np.zeros((self.sequence_length, expected_features))

            # Combine residuals with features
            last_seq = np.column_stack([last_seq.reshape(-1, 1), initial_features])
        else:
            last_seq = last_seq.reshape(-1, 1)
        
        # Predict
        if return_std:
            # Use Monte Carlo approach: make multiple predictions
            n_iterations = 10
            predictions = []

            for _ in range(n_iterations):
                pred_scaled = self.predict(last_seq.copy(), steps, future_feat_array.copy() if future_feat_array is not None else None)
                pred_original = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                predictions.append(pred_original)

            predictions = np.array(predictions)
            mean_pred = predictions.mean(axis=0)
            std_pred = predictions.std(axis=0)

            return pd.Series(mean_pred), std_pred
        else:
            pred_scaled = self.predict(last_seq, steps, future_feat_array)

            # Inverse transform
            pred_original = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

            return pd.Series(pred_original)

    def save(self, filepath: str):
        """
        Save the LSTM model to disk

        Parameters:
        -----------
        filepath : str - Path to save the model (without extension)
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        # Save Keras model
        self.model.save(f"{filepath}.keras")

        # Save metadata and scalers
        metadata = {
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'scaler': self.scaler,
            'feature_scaler': self.feature_scaler,
            'is_fitted': self.is_fitted,
            'last_training_features': self.last_training_features
        }

        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

    @classmethod
    def load(cls, filepath: str) -> 'LSTMModel':
        """
        Load an LSTM model from disk

        Parameters:
        -----------
        filepath : str - Path to load the model from (without extension)

        Returns:
        --------
        LSTMModel - Loaded model instance
        """
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

        # Restore metadata
        instance.scaler = metadata['scaler']
        instance.feature_scaler = metadata['feature_scaler']
        instance.is_fitted = metadata['is_fitted']
        instance.last_training_features = metadata['last_training_features']

        return instance
