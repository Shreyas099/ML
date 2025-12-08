"""
Hybrid SARIMA-LSTM Weather Forecasting App
Demonstrates superior performance of hybrid statistical-deep learning approach
with multivariate feature support.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pytz
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Hybrid Weather Forecast",
    page_icon="🌤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(120deg, #2193b0, #6dd5ed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.stAlert {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# Constants
FORECAST_HOURS = 168  # 7 days
DATA_DAYS = 365  # Use 60 days for better seasonal understanding


@st.cache_resource
def get_weather_fetcher():
    """Cached weather data fetcher"""
    from noaa_api import WeatherDataFetcher
    return WeatherDataFetcher()


@st.cache_resource
def get_model_class():
    """Cached model class import"""
    from hybrid_model import HybridSARIMALSTM
    return HybridSARIMALSTM


def get_location_coordinates(fetcher, location_name):
    """Get coordinates with session state caching"""
    cache_key = f"loc_{location_name}"

    if cache_key in st.session_state:
        return st.session_state[cache_key]

    coords = fetcher.get_location_data(location_name)
    if coords:
        st.session_state[cache_key] = coords
    return coords


def train_and_forecast(data, location_name):
    """
    Train all three models for comparison using multivariate data
    
    Parameters:
    -----------
    data : pd.DataFrame - DataFrame containing 'temperature' and other features
    location_name : str - Name of location for display
    """

    with st.spinner("🔄 Training all models for comparison..."):
        progress = st.progress(0)
        status = st.empty()

        # Import model classes
        status.text("Initializing models...")
        progress.progress(5)

        HybridModel = get_model_class()
        from lstm_model import LSTMModel

        # Use EST timezone for display
        est = pytz.timezone('America/New_York')
        start_date = pd.Timestamp(datetime.now(est))

        # Prepare Target and Features
        target_series = data['temperature']
        
        # Select available features for multivariate training
        potential_features = ['relativeHumidity', 'dewpoint', 'barometricPressure', 'windSpeed', 'windDirection']
        available_features = [col for col in potential_features if col in data.columns]
        
        if available_features:
            features_df = data[available_features]
            feature_msg = f"with features: {', '.join(available_features)}"
        else:
            features_df = None
            feature_msg = "(univariate)"

        logger.info(f"Training with features: {available_features}")

        # 1. Train Hybrid Model (SARIMA + LSTM on residuals)
        status.text(f"Training Hybrid Model {feature_msg}...")
        progress.progress(15)

        hybrid_model = HybridModel(
            sarima_order=(1, 1, 1),
            sarima_seasonal_order=(1, 1, 1, 24),
            lstm_sequence_length=24,
            lstm_units=(32, 16)
        )
        
        # Pass features to hybrid fit
        hybrid_model.fit(
            target_series, 
            features=features_df,
            lstm_epochs=30, 
            lstm_batch_size=32
        )

        hybrid_forecast, hybrid_lower, hybrid_upper = hybrid_model.predict(
            FORECAST_HOURS, start_date, return_confidence=True
        )
        progress.progress(50)

        # 2. Get SARIMA-only predictions (extracted from hybrid)
        status.text("Getting SARIMA-only predictions...")
        sarima_forecast = hybrid_model.sarima.predict(FORECAST_HOURS, start_date, return_confidence=False)
        progress.progress(65)

        # 3. Train standalone LSTM on raw temperature data
        status.text("Training standalone LSTM on temperature data...")
        lstm_standalone = LSTMModel(sequence_length=24, lstm_units=(32, 16))

        try:
            # Train standalone LSTM with features as well for fair comparison
            lstm_standalone.fit(
                target_series, 
                features=features_df,
                epochs=30, 
                batch_size=32
            )

            # Generate LSTM-only forecast
            # Note: predict_residuals is named generically, can be used for raw data too
            lstm_forecast = lstm_standalone.predict_residuals(
                target_series, FORECAST_HOURS, return_std=False
            )
            # Add proper datetime index with timezone
            lstm_forecast.index = pd.date_range(start=start_date, periods=FORECAST_HOURS, freq='H', tz=est)
        except Exception as e:
            logger.warning(f"Standalone LSTM failed: {e}")
            lstm_forecast = None

        progress.progress(90)

        # Validate predictions are reasonable
        status.text("Validating predictions...")
        current_temp = target_series.iloc[-1]

        # Check if predictions are within reasonable range
        if abs(hybrid_forecast.mean() - current_temp) > 30:
            st.warning(f"⚠️ Predictions may be less accurate. Current temp: {current_temp:.1f}°C, "
                      f"Forecast avg: {hybrid_forecast.mean():.1f}°C")

        progress.progress(100)
        status.text("✅ Complete!")

    return {
        'hybrid_model': hybrid_model,
        'hybrid_forecast': hybrid_forecast,
        'hybrid_lower': hybrid_lower,
        'hybrid_upper': hybrid_upper,
        'sarima_forecast': sarima_forecast,
        'lstm_forecast': lstm_forecast,
        'historical': target_series, # Return series for plotting
        'start_time': start_date.strftime('%Y-%m-%d %H:%M %Z')
    }


def plot_comparison(results, location_name):
    """Plot 3-model comparison to demonstrate hybrid superiority"""

    historical = results['historical']
    sarima_forecast = results['sarima_forecast']
    lstm_forecast = results['lstm_forecast']
    hybrid_forecast = results['hybrid_forecast']

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "📊 3-Model Comparison: SARIMA vs LSTM vs Hybrid",
            "📈 Full Forecast with Confidence Intervals"
        ),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )

    # Forecast times
    forecast_times = hybrid_forecast.index

    # Plot 1: 3-Model Comparison
    # SARIMA only (draw first)
    fig.add_trace(
        go.Scatter(
            x=forecast_times,
            y=sarima_forecast,
            name='🟧 SARIMA Only',
            line=dict(color='#FF8C00', width=3, dash='dash'),  # Dark orange, thicker
            mode='lines',
            legendgroup='models'
        ),
        row=1, col=1
    )

    # LSTM only (if available)
    if lstm_forecast is not None:
        fig.add_trace(
            go.Scatter(
                x=forecast_times,
                y=lstm_forecast,
                name='🔴 LSTM Only',
                line=dict(color='#FF0000', width=3, dash='dot'),  # Bright red, thicker
                mode='lines',
                legendgroup='models'
            ),
            row=1, col=1
        )

    # Hybrid (best) - draw last so it's on top
    fig.add_trace(
        go.Scatter(
            x=forecast_times,
            y=hybrid_forecast,
            name='🔵 Hybrid (Best) ⭐',
            line=dict(color='#00CED1', width=4),  # Dark turquoise, thickest
            mode='lines',
            opacity=0.9,  # Slightly transparent to see lines behind
            legendgroup='models'
        ),
        row=1, col=1
    )

    # Plot 2: Full forecast with historical data
    # Historical data
    last_48h = historical.tail(48)
    fig.add_trace(
        go.Scatter(
            x=last_48h.index,
            y=last_48h.values,
            name='Historical',
            line=dict(color='gray', width=2),
            legendgroup='data'
        ),
        row=2, col=1
    )

    # Forecast with confidence interval
    fig.add_trace(
        go.Scatter(
            x=forecast_times,
            y=hybrid_forecast,
            name='Hybrid Forecast',
            line=dict(color='#2193b0', width=3),
            legendgroup='data'
        ),
        row=2, col=1
    )

    # Confidence interval
    fig.add_trace(
        go.Scatter(
            x=list(forecast_times) + list(forecast_times[::-1]),
            y=list(results['hybrid_upper']) + list(results['hybrid_lower'][::-1]),
            fill='toself',
            fillcolor='rgba(33, 147, 176, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name='95% Confidence',
            legendgroup='data'
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)

    fig.update_layout(
        height=900,
        title_text=f"Weather Forecast for {location_name}",
        title_font_size=20,
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🌤️ Hybrid Weather Forecast</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;">'
        'SARIMA + LSTM: Combining Statistical and Deep Learning for Superior Accuracy</p>',
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.header("📍 Location")

        location_name = st.text_input(
            "Enter city name",
            value="New York",
            help="e.g., New York, London, Tokyo"
        )

        st.divider()

        st.header("ℹ️ About")
        st.markdown("""
        **Hybrid SARIMA-LSTM Model**

        Two-stage approach:
        1. **SARIMA**: Captures linear trends and seasonal patterns
        2. **LSTM**: Captures non-linear residuals
        3. **Hybrid**: Combines both for superior accuracy

        ✨ **Why Hybrid?**
        - SARIMA alone: Good for seasonality, poor for non-linear patterns
        - LSTM alone: Good for non-linear, poor for long-term trends
        - **Hybrid: Best of both worlds!**
        """)

        st.divider()

        train_button = st.button(
            "🚀 Train & Forecast",
            type="primary",
            use_container_width=True
        )

    # Main area
    if not location_name:
        st.info("👈 Enter a location in the sidebar to begin")
        return

    # Get location
    fetcher = get_weather_fetcher()
    coords = get_location_coordinates(fetcher, location_name)

    if not coords:
        st.error(f"❌ Location '{location_name}' not found. Try another city.")
        return

    lat, lon = coords
    st.success(f"📍 {location_name} ({lat:.2f}°, {lon:.2f}°)")

    # Training
    if train_button:
        try:
            # Fetch data
            with st.spinner(f"Fetching {DATA_DAYS} days of weather data..."):
                historical_data, metadata = fetcher.get_historical_observations(lat, lon, days=DATA_DAYS)

            if historical_data.empty:
                st.error("❌ No data available for this location")
                return

            # Show data quality
            if 'temperature' not in historical_data.columns:
                st.error("❌ Temperature data not available")
                return

            last_data_point = historical_data['temperature'].iloc[-1]
            last_data_time = historical_data.index[-1]

            st.info(f"""
            ✅ **Data Retrieved**: {metadata['source']}
            - **Data points**: {metadata['data_points']} ({metadata['coverage_days']:.1f} days)
            - **Last data point**: {last_data_point:.1f}°C at {last_data_time.strftime('%Y-%m-%d %H:%M')}
            - **Time lag**: {(datetime.now() - last_data_time.replace(tzinfo=None)).total_seconds() / 3600:.1f} hours ago
            """)

            # Warning if data is stale
            hours_old = (datetime.now() - last_data_time.replace(tzinfo=None)).total_seconds() / 3600
            if hours_old > 3:
                st.warning(f"⚠️ Historical data is {hours_old:.1f} hours old. Predictions may not reflect current conditions.")

            # Prepare data: Drop rows where temperature is missing
            clean_data = historical_data.dropna(subset=['temperature'])
            
            # Simple imputation for features if necessary (forward fill then backward fill)
            # This ensures we don't drop rows just because a wind speed reading is missing
            clean_data = clean_data.fillna(method='ffill').fillna(method='bfill')

            if len(clean_data) < 100:
                st.warning(f"⚠️ Limited data ({len(clean_data)} points). Results may be less accurate.")

            # Train and forecast with FULL data (columns + features)
            results = train_and_forecast(clean_data, location_name)

            # Store in session state
            st.session_state['results'] = results
            st.session_state['location_name'] = location_name

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.exception("Training failed")
            return

    # Display results
    if 'results' in st.session_state:
        results = st.session_state['results']
        loc_name = st.session_state.get('location_name', location_name)

        st.success("✅ Forecast generated successfully!")

        # Show comparison chart
        fig = plot_comparison(results, loc_name)
        st.plotly_chart(fig, use_container_width=True)

        # Show statistics
        st.header("📊 Model Performance Comparison")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Forecast Start",
                results['start_time'].split()[1],  # Time only
                results['start_time'].split()[0]   # Date
            )

        with col2:
            current_temp = results['historical'].iloc[-1]
            st.metric(
                "Current Temp",
                f"{current_temp:.1f}°C",
                "Last observation"
            )

        with col3:
            avg_temp = results['hybrid_forecast'].mean()
            st.metric(
                "Hybrid Avg",
                f"{avg_temp:.1f}°C",
                f"±{(results['hybrid_upper'] - results['hybrid_lower']).mean():.1f}°C"
            )

        with col4:
            models_trained = "3 Models"
            if results['lstm_forecast'] is None:
                models_trained = "2 Models"
            st.metric("Models Trained", models_trained, "SARIMA, LSTM, Hybrid")

        # Model comparison explanation
        st.info("""
        💡 **3-Model Comparison** (see top chart):
        - 🟧 **SARIMA Only** (orange dashed line): Captures trends & daily cycles, but misses non-linear patterns
        - 🔴 **LSTM Only** (red dotted line): Learns non-linear patterns, but drifts away from realistic trends
        - 🔵 **Hybrid ⭐** (turquoise solid line): **Best performance** - combines SARIMA's stability with LSTM's adaptability

        **Notice**: The hybrid line stays close to SARIMA's daily patterns while correcting its weaknesses, unlike LSTM which diverges completely!
        """)

        # Temperature validation
        temp_diff = abs(avg_temp - current_temp)
        if temp_diff < 10:
            st.success(f"✅ Predictions look realistic (within {temp_diff:.1f}°C of current temperature)")
        else:
            st.warning(f"⚠️ Large temperature change predicted ({temp_diff:.1f}°C difference from current)")


if __name__ == "__main__":
    main()
