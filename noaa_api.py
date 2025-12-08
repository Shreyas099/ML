"""
Weather API Data Fetcher (Resilient Version)
Fetches historical and current weather data from Open-Meteo API
Includes automatic retries and backoff for 429 Rate Limits
"""
import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

# Setup logging
logger = logging.getLogger(__name__)


class WeatherDataFetcher:
    """Fetches weather data from Open-Meteo API (free, no API key required)"""

    OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"
    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WeatherForecastApp/2.0',
            'Accept': 'application/json'
        })

    def _fetch_with_retry(self, url: str, params: Dict, retries: int = 3, delay: int = 2) -> Optional[Dict]:
        """
        Helper to fetch data with exponential backoff for 429 errors
        """
        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                
                # If success, return JSON
                if response.status_code == 200:
                    return response.json()
                
                # If rate limited (429), wait and retry
                if response.status_code == 429:
                    wait_time = delay * (2 ** attempt)  # 2s, 4s, 8s...
                    logger.warning(f"Rate limited (429). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                # Other errors
                response.raise_for_status()
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{retries} failed: {e}")
                if attempt == retries - 1:
                    raise e
                time.sleep(delay)
        return None
    
    def get_historical_observations(self, lat: float, lon: float, days: int = 365) -> tuple[pd.DataFrame, dict]:
        """
        Fetch historical weather observations from Open-Meteo
        Combines forecast API (recent 16 days) with archive API (older data)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # For recent data (last 16 days), use forecast API
        recent_start = end_date - timedelta(days=min(16, days))

        # For older data, use archive API
        if days > 16:
            archive_end = end_date - timedelta(days=16)
            archive_start = start_date
        else:
            archive_end = None
            archive_start = None

        metadata = {
            'source': 'unknown',
            'data_points': 0,
            'coverage_days': 0,
            'quality': 'unknown',
            'location': f'{lat:.4f}, {lon:.4f}'
        }

        try:
            dfs = []

            # 1. Get recent data from Forecast API
            logger.info(f"Fetching recent data (last {min(16, days)} days)")
            forecast_params = {
                'latitude': lat,
                'longitude': lon,
                'hourly': 'temperature_2m,relative_humidity_2m,dew_point_2m,pressure_msl,wind_speed_10m,wind_direction_10m',
                'past_days': min(16, days),
                'forecast_days': 0,
                'timezone': 'auto'
            }

            forecast_data = self._fetch_with_retry(self.FORECAST_URL, forecast_params)

            if forecast_data:
                hourly = forecast_data.get('hourly', {})
                times = pd.to_datetime(hourly.get('time', []))

                recent_df = pd.DataFrame({
                    'timestamp': times,
                    'temperature': hourly.get('temperature_2m', []),
                    'dewpoint': hourly.get('dew_point_2m', []),
                    'barometricPressure': hourly.get('pressure_msl', []),
                    'windSpeed': hourly.get('wind_speed_10m', []),
                    'windDirection': hourly.get('wind_direction_10m', []),
                    'relativeHumidity': hourly.get('relative_humidity_2m', [])
                })

                if not recent_df.empty:
                    recent_df = recent_df.set_index('timestamp')
                    dfs.append(recent_df)

            # 2. Get older data from Archive API
            if archive_start is not None and days > 16:
                logger.info(f"Fetching archive data...")
                archive_params = {
                    'latitude': lat,
                    'longitude': lon,
                    'start_date': archive_start.strftime('%Y-%m-%d'),
                    'end_date': archive_end.strftime('%Y-%m-%d'),
                    'hourly': 'temperature_2m,relative_humidity_2m,dew_point_2m,pressure_msl,wind_speed_10m,wind_direction_10m',
                    'timezone': 'auto'
                }

                archive_data = self._fetch_with_retry(self.OPENMETEO_URL, archive_params)

                if archive_data:
                    hourly = archive_data.get('hourly', {})
                    times = pd.to_datetime(hourly.get('time', []))

                    archive_df = pd.DataFrame({
                        'timestamp': times,
                        'temperature': hourly.get('temperature_2m', []),
                        'dewpoint': hourly.get('dew_point_2m', []),
                        'barometricPressure': hourly.get('pressure_msl', []),
                        'windSpeed': hourly.get('wind_speed_10m', []),
                        'windDirection': hourly.get('wind_direction_10m', []),
                        'relativeHumidity': hourly.get('relative_humidity_2m', [])
                    })

                    if not archive_df.empty:
                        archive_df = archive_df.set_index('timestamp')
                        dfs.append(archive_df)

            # Combine dataframes
            if dfs:
                df = pd.concat(dfs).sort_index()
                df = df[~df.index.duplicated(keep='last')] # Dedup

                metadata['source'] = "Open-Meteo API"
                metadata['data_points'] = len(df)
                coverage = (df.index[-1] - df.index[0]).total_seconds() / 86400
                metadata['coverage_days'] = round(coverage, 1)

                if metadata['data_points'] >= 1000:
                    metadata['quality'] = 'good'
                else:
                    metadata['quality'] = 'limited'

                logger.info(f"Success! Got {len(df)} rows.")
                return df, metadata
            else:
                raise ValueError("No data returned from APIs")

        except Exception as e:
            logger.error(f"API Error: {e}. Switching to synthetic data.")
            # Fallback to synthetic
            df = self._generate_synthetic_data(start_date, end_date)
            metadata['source'] = f'Synthetic (Error: {str(e)[:50]})'
            metadata['data_points'] = len(df)
            metadata['quality'] = 'synthetic'
            return df, metadata
    
    def _generate_synthetic_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate synthetic weather data for demonstration when API data is limited"""
        import numpy as np
        
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
        n = len(dates)
        
        # Generate realistic weather patterns with seasonality
        t = np.arange(n)
        seasonal_temp = 20 + 10 * np.sin(2 * np.pi * t / (365.25 * 24))  # Annual cycle
        daily_temp = 5 * np.sin(2 * np.pi * t / 24)  # Daily cycle
        noise = np.random.normal(0, 2, n)
        temperature = seasonal_temp + daily_temp + noise
        
        dewpoint = temperature - np.random.uniform(2, 8, n)
        wind_speed = np.random.uniform(5, 25, n)
        pressure = 1013 + np.random.normal(0, 5, n)
        humidity = 50 + 30 * np.sin(2 * np.pi * t / (365.25 * 24)) + np.random.normal(0, 10, n)
        humidity = np.clip(humidity, 0, 100)
        visibility = 10 + np.random.normal(0, 2, n)
        visibility = np.clip(visibility, 0, 20)
        
        df = pd.DataFrame({
            'temperature': temperature,
            'dewpoint': dewpoint,
            'windSpeed': wind_speed,
            'barometricPressure': pressure,
            'relativeHumidity': humidity,
            'visibility': visibility
        }, index=dates)
        
        return df
    
    def get_current_forecast(self, lat: float, lon: float) -> Dict:
        """Get current forecast for a location using Open-Meteo"""
        # Note: Not actively used in the main training loop, but useful for extensions
        return {}

    def get_location_data(self, location_name: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a location name using Open-Meteo Geocoding API"""
        try:
            params = {'name': location_name, 'count': 1, 'language': 'en', 'format': 'json'}
            response = self.session.get(self.GEOCODING_URL, params=params, timeout=10)
            if response.status_code == 200:
                results = response.json().get('results', [])
                if results:
                    return (results[0]['latitude'], results[0]['longitude'])
            return None
        except:
            return None
