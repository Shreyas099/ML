"""
Weather API Data Fetcher
Fetches historical and current weather data from Open-Meteo API
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
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WeatherForecastApp/2.0',
            'Accept': 'application/json'
        })
    
    def get_historical_observations(self, lat: float, lon: float, days: int = 365) -> tuple[pd.DataFrame, dict]:
        """
        Fetch historical weather observations from Open-Meteo
        Combines forecast API (recent 16 days) with archive API (older data)

        Parameters:
        -----------
        lat : float - Latitude
        lon : float - Longitude
        days : int - Number of days of historical data (default 365)

        Returns:
        --------
        tuple: (DataFrame, metadata_dict)
            - DataFrame with weather observations
            - metadata_dict with keys: 'source', 'data_points', 'coverage_days', 'quality'
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # For recent data (last 16 days), use forecast API which includes past observations
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

            # 1. Get recent data from Forecast API (includes past 16 days + forecast)
            logger.info(f"Fetching recent data from forecast API (last {min(16, days)} days)")
            forecast_params = {
                'latitude': lat,
                'longitude': lon,
                'hourly': 'temperature_2m,relative_humidity_2m,dew_point_2m,pressure_msl,wind_speed_10m,wind_direction_10m',
                'past_days': min(16, days),
                'forecast_days': 0,  # Don't need forecast, just past
                'timezone': 'auto'
            }

            forecast_response = self.session.get('https://api.open-meteo.com/v1/forecast',
                                                params=forecast_params, timeout=30)
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()

            # Parse forecast API data
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
                logger.info(f"Got {len(recent_df)} recent data points")

            # 2. Get older data from Archive API if needed
            if archive_start is not None and days > 16:
                logger.info(f"Fetching archive data from {archive_start.strftime('%Y-%m-%d')} to {archive_end.strftime('%Y-%m-%d')}")
                archive_params = {
                    'latitude': lat,
                    'longitude': lon,
                    'start_date': archive_start.strftime('%Y-%m-%d'),
                    'end_date': archive_end.strftime('%Y-%m-%d'),
                    'hourly': 'temperature_2m,relative_humidity_2m,dew_point_2m,pressure_msl,wind_speed_10m,wind_direction_10m',
                    'timezone': 'auto'
                }

                archive_response = self.session.get(self.OPENMETEO_URL, params=archive_params, timeout=30)
                archive_response.raise_for_status()
                archive_data = archive_response.json()

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
                    logger.info(f"Got {len(archive_df)} archive data points")

            # Combine dataframes
            if dfs:
                df = pd.concat(dfs).sort_index()
                # Remove duplicates, keeping the most recent data (from forecast API)
                df = df[~df.index.duplicated(keep='last')]

                # Update metadata
                sources_used = []
                if len(dfs) == 2:
                    sources_used = ["Forecast API (recent)", "Archive API (historical)"]
                elif dfs and len(recent_df) > 0:
                    sources_used = ["Forecast API (up-to-date)"]
                else:
                    sources_used = ["Archive API"]

                metadata['source'] = f"Open-Meteo: {', '.join(sources_used)}"
                metadata['data_points'] = len(df)
                coverage = (df.index[-1] - df.index[0]).total_seconds() / 86400
                metadata['coverage_days'] = round(coverage, 1)

                # Assess data quality
                if metadata['data_points'] >= 1000 and coverage >= 30:
                    metadata['quality'] = 'good'
                elif metadata['data_points'] >= 200:
                    metadata['quality'] = 'limited'
                else:
                    metadata['quality'] = 'insufficient'

                logger.info(f"Combined {metadata['data_points']} data points ({metadata['coverage_days']} days coverage)")
                logger.info(f"Data range: {df.index[0]} to {df.index[-1]}")
                return df, metadata
            else:
                # Fallback to synthetic data
                df = self._generate_synthetic_data(start_date, end_date)
                metadata['source'] = 'Synthetic (Open-Meteo returned no data)'
                metadata['data_points'] = len(df)
                metadata['coverage_days'] = days
                metadata['quality'] = 'synthetic'
                return df, metadata

        except Exception as e:
            logger.error(f"Error fetching from Open-Meteo: {e}")
            # Return synthetic data as fallback
            df = self._generate_synthetic_data(start_date, end_date)
            metadata['source'] = f'Synthetic (Error: {str(e)[:50]})'
            metadata['data_points'] = len(df)
            metadata['coverage_days'] = days
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
        try:
            params = {
                'latitude': lat,
                'longitude': lon,
                'hourly': 'temperature_2m,precipitation_probability,weather_code',
                'forecast_days': 7,
                'timezone': 'auto'
            }

            response = self.session.get('https://api.open-meteo.com/v1/forecast', params=params, timeout=10)
            response.raise_for_status()
            forecast_data = response.json()

            return forecast_data
        except Exception as e:
            logger.error(f"Error getting forecast: {e}")
            return {}

    def get_location_data(self, location_name: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a location name using Open-Meteo Geocoding API"""
        try:
            # Try the full location name first
            params = {
                'name': location_name,
                'count': 1,
                'language': 'en',
                'format': 'json'
            }

            response = self.session.get(self.GEOCODING_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = data.get('results', [])

            # If no results and location has comma (e.g., "New York, NY"), try just the city name
            if not results and ',' in location_name:
                city_only = location_name.split(',')[0].strip()
                logger.info(f"No results for '{location_name}', trying '{city_only}'")
                params['name'] = city_only
                response = self.session.get(self.GEOCODING_URL, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                results = data.get('results', [])

            if results:
                location = results[0]
                lat = location.get('latitude')
                lon = location.get('longitude')
                name = location.get('name', location_name)
                country = location.get('country', '')
                admin1 = location.get('admin1', '')
                full_name = f"{name}, {admin1}, {country}" if admin1 and country else name
                logger.info(f"Found location: {full_name} at ({lat}, {lon})")
                return (lat, lon)
            else:
                logger.warning(f"No location found for: {location_name}")
                return None
        except Exception as e:
            logger.error(f"Error geocoding location: {e}")
            return None
