"""
Streamlit Weather Forecasting App
Hybrid SARIMA-LSTM Model for 7-Day Weather Predictions
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Constants
FORECAST_HOURS = 168  # 7 days * 24 hours

# Lazy imports to speed up startup - don't load until actually needed
_models_loaded = False
_NOAADataFetcher = None
_HybridSARIMALSTM = None

def get_models():
    """Lazy load models to speed up app startup - only when actually needed"""
    global _models_loaded, _NOAADataFetcher, _HybridSARIMALSTM

    if not _models_loaded:
        try:
            from noaa_api import WeatherDataFetcher
            _NOAADataFetcher = WeatherDataFetcher  # Keep variable name for compatibility
            # Don't import hybrid_model yet - it will import TensorFlow
            _HybridSARIMALSTM = None  # Will be loaded on demand
        except Exception as e:
            # Don't use st.error here - Streamlit might not be initialized yet
            logger.error(f"Error loading data fetcher: {e}")
            return None, None
        _models_loaded = True

    return _NOAADataFetcher, _HybridSARIMALSTM

def get_hybrid_model():
    """Load hybrid model only when training is requested"""
    global _HybridSARIMALSTM
    if _HybridSARIMALSTM is None:
        try:
            from hybrid_model import HybridSARIMALSTM
            _HybridSARIMALSTM = HybridSARIMALSTM
        except Exception as e:
            # Error will be shown in UI when this is called
            logger.error(f"Error loading hybrid model: {e}")
            return None
    return _HybridSARIMALSTM

# Page configuration
st.set_page_config(
    page_title="Weather Forecast App",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = {}
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}

# Initialize API fetcher (lazy loaded)
def get_fetcher():
    NOAADataFetcher, _ = get_models()
    if NOAADataFetcher is None:
        return None
    return NOAADataFetcher()

def main():
    # Header - render immediately to pass health check
    st.markdown('<h1 class="main-header">🌤️ Hybrid Weather Forecast</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">SARIMA-LSTM Model for 7-Day Weather Predictions</p>', unsafe_allow_html=True)
    
    # Don't check models here - let it fail gracefully when actually used
    
    # Sidebar
    with st.sidebar:
        st.header("📍 Location Settings")
        
        # Location input method
        input_method = st.radio(
            "Select Input Method",
            ["City Name", "Coordinates"],
            help="Choose how to specify the location"
        )
        
        if input_method == "City Name":
            location_name = st.text_input(
                "Enter City Name",
                value="New York, NY",
                help="e.g., New York, NY or London, UK"
            )
            lat, lon = None, None
            
            if location_name:
                fetcher = get_fetcher()
                if fetcher is None:
                    st.error("Unable to initialize data fetcher")
                    st.stop()
                with st.spinner("Finding location..."):
                    coords = fetcher.get_location_data(location_name)
                    if coords:
                        lat, lon = coords
                        st.success(f"📍 {location_name}")
                        st.info(f"Lat: {lat:.4f}, Lon: {lon:.4f}")
                    else:
                        st.error("Location not found. Please try again.")
                        st.stop()
        else:
            # Input validation for coordinates
            lat = st.number_input("Latitude", value=40.7128, min_value=-90.0, max_value=90.0, format="%.4f")
            lon = st.number_input("Longitude", value=-74.0060, min_value=-180.0, max_value=180.0, format="%.4f")
            location_name = f"({lat:.4f}, {lon:.4f})"
        
        st.divider()
        
        # Model parameters
        st.header("⚙️ Model Parameters")
        
        st.subheader("SARIMA")
        sarima_p = st.slider("AR Order (p)", 0, 3, 1)
        sarima_d = st.slider("Differencing (d)", 0, 2, 1)
        sarima_q = st.slider("MA Order (q)", 0, 3, 1)
        seasonal_period = st.selectbox("Seasonal Period", [24, 168], index=0, 
                                      help="24 for daily, 168 for weekly")
        
        st.subheader("LSTM")
        lstm_sequence = st.slider("Sequence Length", 10, 60, 30)
        lstm_units_1 = st.slider("LSTM Layer 1 Units", 16, 128, 64)
        lstm_units_2 = st.slider("LSTM Layer 2 Units", 8, 64, 32)
        lstm_epochs = st.slider("Training Epochs", 10, 100, 50)
        
        st.divider()
        
        # Training button
        train_button = st.button("🚀 Train Model & Generate Forecast", type="primary", use_container_width=True)
    
    # Main content area
    if lat is None or lon is None:
        st.info("👈 Please configure location in the sidebar")
        return
    
    # Create location key
    location_key = f"{lat:.4f}_{lon:.4f}"
    
    # Training and prediction
    if train_button:
        with st.spinner("Fetching weather data..."):
            fetcher = get_fetcher()
            if fetcher is None:
                st.error("Unable to initialize data fetcher")
                st.stop()

            # Fetch historical data directly with coordinates (Open-Meteo doesn't need station IDs)
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Fetching historical weather data from Open-Meteo...")
            progress_bar.progress(20)

            # Request 365 days of data (Open-Meteo can provide this!)
            historical_data, data_metadata = fetcher.get_historical_observations(lat, lon, days=365)

            if historical_data.empty:
                st.error("❌ Failed to fetch data. Please try again.")
                return

            # Display data source information
            if data_metadata['quality'] == 'synthetic':
                st.warning(f"⚠️ **Using Synthetic Data**: {data_metadata['source']}")
                st.info("📊 Synthetic data is used for demonstration only. Results may not reflect real weather patterns.")
            elif data_metadata['quality'] == 'insufficient':
                st.warning(f"⚠️ **Limited Data**: Only {data_metadata['data_points']} data points covering {data_metadata['coverage_days']} days")
                st.info("For best results, at least 1000 data points covering 30+ days are recommended.")
            elif data_metadata['quality'] == 'limited':
                st.info(f"📡 **Data Source**: {data_metadata['source']} - {data_metadata['data_points']} points, {data_metadata['coverage_days']} days coverage")
            else:
                st.success(f"✅ **Good Data Quality**: {data_metadata['source']} - {data_metadata['data_points']} points, {data_metadata['coverage_days']} days coverage")
            
            progress_bar.progress(40)
            status_text.text("Preparing data for training...")

            # Prepare training data
            # Use temperature as primary target
            if 'temperature' in historical_data.columns:
                temp_data = historical_data['temperature'].dropna()
            else:
                st.error("Temperature data not available")
                return

            # Validate data length
            if len(temp_data) < 100:
                st.error(f"❌ Insufficient data: Only {len(temp_data)} temperature readings available. Minimum 100 required.")
                return
            elif len(temp_data) < 200:
                st.warning(f"⚠️ Limited data ({len(temp_data)} points). Results may be inaccurate. Recommended: 1000+ points.")
            
            # Prepare features
            feature_cols = ['dewpoint', 'barometricPressure', 'windSpeed', 'visibility']
            available_features = [col for col in feature_cols if col in historical_data.columns]
            
            if available_features:
                features_df = historical_data[available_features].dropna()
                # Align with temperature data
                common_idx = temp_data.index.intersection(features_df.index)
                if len(common_idx) > 100:
                    temp_data = temp_data.loc[common_idx]
                    features_df = features_df.loc[common_idx]
                else:
                    features_df = None
            else:
                features_df = None
            
            progress_bar.progress(60)
            status_text.text("Training hybrid SARIMA-LSTM model...")
            
            # Initialize and train model (lazy loaded - only now import TensorFlow)
            HybridSARIMALSTM_class = get_hybrid_model()
            if HybridSARIMALSTM_class is None:
                st.error("Unable to load model classes. TensorFlow may be initializing...")
                return
            model = HybridSARIMALSTM_class(
                sarima_order=(sarima_p, sarima_d, sarima_q),
                sarima_seasonal_order=(1, 1, 1, seasonal_period),
                lstm_sequence_length=lstm_sequence,
                lstm_units=(lstm_units_1, lstm_units_2)
            )
            
            # Train model
            model.fit(
                temp_data,
                features_df,
                lstm_epochs=lstm_epochs,
                lstm_batch_size=32
            )
            
            progress_bar.progress(80)
            status_text.text("Generating 7-day forecast...")

            # Generate 7-day forecast (168 hours) with confidence intervals
            start_date = datetime.now()
            # Note: future_features would need to be forecasted separately
            # For now, pass None as we don't have future feature values
            forecast, forecast_lower, forecast_upper = model.predict(
                FORECAST_HOURS, pd.Timestamp(start_date), None, return_confidence=True
            )

            # Store in session state
            st.session_state.models[location_key] = model
            st.session_state.forecasts[location_key] = {
                'forecast': forecast,
                'lower': forecast_lower,
                'upper': forecast_upper
            }
            st.session_state.data_cache[location_key] = {
                'historical': temp_data,
                'features': features_df,
                'location': location_name
            }
            
            progress_bar.progress(100)
            status_text.text("✅ Forecast generated successfully!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
    
    # Display results
    if location_key in st.session_state.forecasts:
        forecast_data = st.session_state.forecasts[location_key]
        forecast = forecast_data['forecast']
        forecast_lower = forecast_data['lower']
        forecast_upper = forecast_data['upper']
        historical = st.session_state.data_cache[location_key]['historical']
        location_display = st.session_state.data_cache[location_key]['location']

        st.success(f"✅ Forecast available for {location_display}")

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Current Temp", f"{historical.iloc[-1]:.1f}°C" if len(historical) > 0 else "N/A")
        with col2:
            st.metric("Forecast Start", forecast.index[0].strftime("%Y-%m-%d"))
        with col3:
            st.metric("Forecast End", forecast.index[-1].strftime("%Y-%m-%d"))
        with col4:
            avg_forecast = forecast.mean()
            st.metric("Avg Forecast", f"{avg_forecast:.1f}°C")
        
        st.divider()
        
        # Forecast visualization
        st.header("📊 7-Day Temperature Forecast")
        
        # Create interactive plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Temperature Forecast", "Forecast Details"),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
        
        # Historical data (last 7 days or available data)
        hist_recent = historical.tail(min(FORECAST_HOURS, len(historical)))
        
        # Forecast plot
        fig.add_trace(
            go.Scatter(
                x=hist_recent.index,
                y=hist_recent.values,
                name="Historical",
                line=dict(color="#667eea", width=2),
                mode="lines"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast.values,
                name="Forecast",
                line=dict(color="#f093fb", width=2, dash="dash"),
                mode="lines"
            ),
            row=1, col=1
        )

        # Add proper confidence intervals (95%)
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast_upper.values,
                name="Upper 95% CI",
                line=dict(width=0),
                mode="lines",
                showlegend=False
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast_lower.values,
                name="95% Confidence Interval",
                fill="tonexty",
                fillcolor="rgba(246, 147, 251, 0.2)",
                line=dict(width=0),
                mode="lines"
            ),
            row=1, col=1
        )
        
        # Daily averages
        forecast_daily = forecast.resample('D').mean()
        fig.add_trace(
            go.Bar(
                x=forecast_daily.index,
                y=forecast_daily.values,
                name="Daily Average",
                marker_color="#764ba2",
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Avg Temp (°C)", row=2, col=1)
        fig.update_layout(
            height=700,
            showlegend=True,
            hovermode="x unified",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast table
        st.subheader("📋 Detailed Forecast Table")
        
        # Create daily summary
        forecast_df = pd.DataFrame({
            'Date': forecast.index.date,
            'Time': forecast.index.time,
            'Temperature (°C)': forecast.values
        })
        
        # Daily summary
        daily_summary = forecast_df.groupby('Date').agg({
            'Temperature (°C)': ['mean', 'min', 'max']
        }).round(2)
        daily_summary.columns = ['Avg Temp', 'Min Temp', 'Max Temp']
        daily_summary = daily_summary.reset_index()
        daily_summary['Date'] = pd.to_datetime(daily_summary['Date'])
        daily_summary['Date'] = daily_summary['Date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(daily_summary, use_container_width=True, hide_index=True)
        
        # Download button
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Forecast (CSV)",
            data=csv,
            file_name=f"weather_forecast_{location_key}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    else:
        # Welcome message
        st.info("👈 Click 'Train Model & Generate Forecast' in the sidebar to get started!")
        
        # Show example locations
        st.subheader("🌍 Example Locations")
        example_locations = [
            ("New York, NY", 40.7128, -74.0060),
            ("Los Angeles, CA", 34.0522, -118.2437),
            ("Chicago, IL", 41.8781, -87.6298),
            ("London, UK", 51.5074, -0.1278),
            ("Tokyo, Japan", 35.6762, 139.6503)
        ]
        
        cols = st.columns(len(example_locations))
        for i, (name, lat_ex, lon_ex) in enumerate(example_locations):
            with cols[i]:
                st.write(f"**{name}**")
                st.caption(f"Lat: {lat_ex}, Lon: {lon_ex}")

# Ensure main() is called - Streamlit Cloud expects this
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Fallback error display if main() fails
        st.error(f"Application error: {str(e)}")
        st.info("Please check the logs for more details.")
        import traceback
        st.code(traceback.format_exc())
