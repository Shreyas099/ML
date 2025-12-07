"""
NOAA Weather API Data Fetcher
Fetches historical and current weather data from NOAA API
"""
import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

# Setup logging
logger = logging.getLogger(__name__)


class NOAADataFetcher:
    """Fetches weather data from NOAA Weather API"""
    
    BASE_URL = "https://api.weather.gov"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WeatherForecastApp/1.0 (weather-forecast-app)',
            'Accept': 'application/json'
        })
    
    def get_station_id(self, lat: float, lon: float) -> Optional[str]:
        """Get the nearest weather station ID for given coordinates"""
        try:
            url = f"{self.BASE_URL}/points/{lat},{lon}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Get observation stations
            stations_url = data['properties']['observationStations']
            stations_response = self.session.get(stations_url, timeout=10)
            stations_response.raise_for_status()
            stations_data = stations_response.json()
            
            if stations_data['features']:
                return stations_data['features'][0]['properties']['stationIdentifier']
            return None
        except Exception as e:
            logger.error(f"Error getting station ID: {e}")
            return None
    
    def get_historical_observations(self, station_id: str, days: int = 30) -> tuple[pd.DataFrame, dict]:
        """
        Fetch historical observations from a station
        Note: NOAA API has limited historical data, so we'll use current observations
        and simulate historical data structure for training

        Returns:
        --------
        tuple: (DataFrame, metadata_dict)
            - DataFrame with weather observations
            - metadata_dict with keys: 'source', 'data_points', 'coverage_days', 'quality'
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        observations = []
        metadata = {
            'source': 'unknown',
            'data_points': 0,
            'coverage_days': 0,
            'quality': 'unknown',
            'station_id': station_id
        }

        # NOAA API doesn't provide extensive historical data via free API
        # We'll fetch recent observations and use them for training
        # In production, you might want to use a different data source for historical data

        try:
            # Get recent observations (last 7 days typically available)
            url = f"{self.BASE_URL}/stations/{station_id}/observations"
            params = {
                'limit': 500,
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            for feature in data.get('features', []):
                props = feature['properties']
                timestamp = props.get('timestamp')
                if timestamp:
                    obs = {
                        'timestamp': pd.to_datetime(timestamp),
                        'temperature': props.get('temperature', {}).get('value'),
                        'dewpoint': props.get('dewpoint', {}).get('value'),
                        'windSpeed': props.get('windSpeed', {}).get('value'),
                        'windDirection': props.get('windDirection', {}).get('value'),
                        'barometricPressure': props.get('barometricPressure', {}).get('value'),
                        'relativeHumidity': props.get('relativeHumidity', {}).get('value'),
                        'visibility': props.get('visibility', {}).get('value'),
                        'precipitationLastHour': props.get('precipitationLastHour', {}).get('value')
                    }
                    observations.append(obs)

            df = pd.DataFrame(observations)
            if not df.empty:
                df = df.sort_values('timestamp')
                df = df.set_index('timestamp')

                # Update metadata for real data
                metadata['source'] = 'NOAA API (real data)'
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

                return df, metadata
            else:
                # Generate synthetic data for demonstration if no data available
                df = self._generate_synthetic_data(start_date, end_date)
                metadata['source'] = 'Synthetic (NOAA unavailable)'
                metadata['data_points'] = len(df)
                metadata['coverage_days'] = days
                metadata['quality'] = 'synthetic'
                return df, metadata

        except Exception as e:
            logger.error(f"Error fetching observations: {e}")
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
        """Get current forecast for a location"""
        try:
            url = f"{self.BASE_URL}/points/{lat},{lon}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            forecast_url = data['properties']['forecast']
            forecast_response = self.session.get(forecast_url, timeout=10)
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()
            
            return forecast_data
        except Exception as e:
            logger.error(f"Error getting forecast: {e}")
            return {}

    def get_location_data(self, location_name: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a location name using geocoding"""
        from geopy.geocoders import Nominatim

        try:
            geolocator = Nominatim(user_agent="weather_app", timeout=10)
            location = geolocator.geocode(location_name)
            if location:
                return (location.latitude, location.longitude)
        except Exception as e:
            logger.error(f"Error geocoding location: {e}")
        return None
