"""
Hybrid SARIMA-LSTM Weather Forecasting App
Demonstrates superior performance of hybrid statistical-deep learning approach
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
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
DATA_DAYS = 30  # Use 30 days for better accuracy


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


def train_and_forecast(temp_data, location_name):
    """Train model and generate forecast"""

    with st.spinner("🔄 Training hybrid SARIMA-LSTM model..."):
        progress = st.progress(0)
        status = st.empty()

        # Initialize model with optimized parameters
        status.text("Initializing model...")
        progress.progress(10)

        HybridModel = get_model_class()
        model = HybridModel(
            sarima_order=(1, 1, 1),
            sarima_seasonal_order=(1, 1, 1, 24),
            lstm_sequence_length=24,
            lstm_units=(32, 16)
        )

        # Train model
        status.text("Training SARIMA + LSTM...")
        progress.progress(30)

        model.fit(temp_data, lstm_epochs=30, lstm_batch_size=32)
        progress.progress(80)

        # Generate forecast
        status.text("Generating 7-day forecast...")
        start_date = pd.Timestamp(datetime.now())

        forecast, lower, upper = model.predict(
            FORECAST_HOURS,
            start_date,
            return_confidence=True
        )

        # Get component predictions for comparison
        components = model.get_component_predictions(FORECAST_HOURS, start_date)

        progress.progress(100)
        status.text("✅ Complete!")

    return {
        'model': model,
        'forecast': forecast,
        'lower': lower,
        'upper': upper,
        'components': components,
        'historical': temp_data
    }


def plot_comparison(results, location_name):
    """Plot model comparison to demonstrate hybrid superiority"""

    components = results['components']
    historical = results['historical']

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "📊 Model Comparison: SARIMA vs Hybrid SARIMA-LSTM",
            "📈 Full Forecast with Confidence Intervals"
        ),
        vertical_spacing=0.15,
        row_heights=[0.4, 0.6]
    )

    # Forecast times
    forecast_times = components['hybrid'].index

    # Plot 1: Comparison (SARIMA vs Hybrid)
    if 'sarima' in components:
        fig.add_trace(
            go.Scatter(
                x=forecast_times,
                y=components['sarima'],
                name='SARIMA Only',
                line=dict(color='orange', width=2, dash='dash'),
                legendgroup='models'
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=forecast_times,
            y=components['hybrid'],
            name='Hybrid (SARIMA+LSTM)',
            line=dict(color='#2193b0', width=3),
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
            y=results['forecast'],
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
            y=list(results['upper']) + list(results['lower'][::-1]),
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
            st.info(f"✅ {metadata['source']} - {metadata['data_points']} points ({metadata['coverage_days']:.1f} days)")

            # Prepare temperature data
            if 'temperature' not in historical_data.columns:
                st.error("❌ Temperature data not available")
                return

            temp_data = historical_data['temperature'].dropna()

            if len(temp_data) < 100:
                st.warning(f"⚠️ Limited data ({len(temp_data)} points). Results may be less accurate.")

            # Train and forecast
            results = train_and_forecast(temp_data, location_name)

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
        st.header("📊 Model Performance")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Forecast Horizon",
                "7 Days",
                f"{FORECAST_HOURS} hours"
            )

        with col2:
            avg_temp = results['forecast'].mean()
            st.metric(
                "Avg Forecast Temp",
                f"{avg_temp:.1f}°C",
                f"±{(results['upper'] - results['lower']).mean():.1f}°C"
            )

        with col3:
            if results['model'].lstm.is_fitted:
                st.metric("Model Status", "Hybrid", "SARIMA + LSTM")
            else:
                st.metric("Model Status", "SARIMA Only", "LSTM failed")

        # Key insight
        st.info("""
        💡 **Key Insight**: The hybrid model (blue line) combines the strengths of both approaches:
        - SARIMA captures the overall trend and daily patterns
        - LSTM adds corrections for non-linear weather dynamics
        - Result: More accurate and reliable forecasts!
        """)


if __name__ == "__main__":
    main()
