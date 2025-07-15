import numpy as np
import pandas as pd

def tsv_to_temperature_estimate(tsv_values, clothing_level=1.0, activity_level=1.2, humidity=50, air_velocity=0.1):
    """
    Convert TSV (Thermal Sensation Vote) to estimated temperature using thermal comfort models.
    
    Parameters:
    - tsv_values: Array of TSV values (-3 to +3)
    - clothing_level: Clothing insulation level (clo), default 1.0 (typical indoor clothing)
    - activity_level: Metabolic rate (met), default 1.2 (seated, light activity)
    - humidity: Relative humidity (%), default 50%
    - air_velocity: Air velocity (m/s), default 0.1
    
    Returns:
    - Estimated temperatures in degrees Celsius
    """
    
    # Convert inputs to numpy arrays and ensure they're 1D
    tsv_values = np.asarray(tsv_values).flatten()
    clothing_level = np.asarray(clothing_level).flatten()
    activity_level = np.asarray(activity_level).flatten()
    humidity = np.asarray(humidity).flatten()
    air_velocity = np.asarray(air_velocity).flatten()
    
    # Ensure all arrays have the same length
    n_samples = len(tsv_values)
    if len(clothing_level) == 1:
        clothing_level = np.full(n_samples, clothing_level[0])
    if len(activity_level) == 1:
        activity_level = np.full(n_samples, activity_level[0])
    if len(humidity) == 1:
        humidity = np.full(n_samples, humidity[0])
    if len(air_velocity) == 1:
        air_velocity = np.full(n_samples, air_velocity[0])
    
    # Base comfort temperature (neutral TSV = 0)
    # This varies by climate and season, but typically around 22-24°C for indoor environments
    base_comfort_temp = 23.0  # degrees Celsius
    
    # TSV to temperature conversion factors based on thermal comfort research
    # Each TSV unit typically corresponds to 2-3°C change in temperature
    tsv_temp_factor = 2.5  # degrees Celsius per TSV unit
    
    # Convert TSV to temperature estimate
    temperature_estimates = base_comfort_temp + (tsv_values * tsv_temp_factor)
    
    # Apply clothing and activity adjustments
    # Higher clothing insulation reduces the temperature needed for comfort
    clothing_adjustment = (clothing_level - 1.0) * -2.0  # -2°C per additional clo unit
    
    # Higher activity levels increase the temperature needed for comfort
    activity_adjustment = (activity_level - 1.2) * 3.0  # +3°C per additional met unit
    
    # Humidity adjustment (higher humidity feels warmer)
    humidity_adjustment = (humidity - 50) * 0.02  # +0.02°C per % humidity above 50%
    
    # Air velocity adjustment (higher air velocity feels cooler)
    air_velocity_adjustment = (air_velocity - 0.1) * -5.0  # -5°C per m/s increase
    
    # Apply all adjustments
    final_temperatures = (temperature_estimates + 
                         clothing_adjustment + 
                         activity_adjustment + 
                         humidity_adjustment + 
                         air_velocity_adjustment)
    
    return final_temperatures

def estimate_temperature_from_features(df, tsv_predictions):
    """
    Estimate temperature from features and TSV predictions using more sophisticated thermal comfort models.
    
    Parameters:
    - df: DataFrame with features like CLO_score, MET_score, HumidityPerception, etc.
    - tsv_predictions: Array of predicted TSV values
    
    Returns:
    - Estimated temperatures in degrees Celsius
    """
    
    # Convert tsv_predictions to numpy array
    tsv_predictions = np.asarray(tsv_predictions).flatten()
    
    # Extract relevant features and convert to numpy arrays
    clothing_level = df.get('CLO_score', 1.0).fillna(1.0).values
    activity_level = df.get('MET_score', 1.2).fillna(1.2).values / 58.2  # Convert W/m² to met
    
    # Humidity perception to relative humidity estimate
    humidity_perception = df.get('HumidityPerception', 0).fillna(0).values
    # Convert perception scale (-3 to +3) to relative humidity (30% to 80%)
    humidity = 55 + (humidity_perception * 8.33)  # 55% ± 25%
    
    # Air movement perception to air velocity estimate
    air_movement = df.get('AirMovement', 0).fillna(0).values
    # Convert perception scale (-2 to +3) to air velocity (0.05 to 0.3 m/s)
    air_velocity = 0.1 + (air_movement * 0.05)
    
    # Calculate temperature estimates
    temperatures = tsv_to_temperature_estimate(
        tsv_predictions,
        clothing_level=clothing_level,
        activity_level=activity_level,
        humidity=humidity,
        air_velocity=air_velocity
    )
    
    return temperatures

def comfort_range_to_temperature(comfort_level):
    """
    Convert comfort level categories to typical temperature ranges.
    
    Parameters:
    - comfort_level: String like "Cold", "Comfortable", "Hot"
    
    Returns:
    - Dictionary with min, max, and typical temperature
    """
    
    comfort_ranges = {
        "Very Cold": {"min": 10, "max": 15, "typical": 12.5},
        "Cold": {"min": 15, "max": 18, "typical": 16.5},
        "Cool": {"min": 18, "max": 21, "typical": 19.5},
        "Comfortable": {"min": 21, "max": 25, "typical": 23.0},
        "Warm": {"min": 25, "max": 28, "typical": 26.5},
        "Hot": {"min": 28, "max": 32, "typical": 30.0},
        "Very Hot": {"min": 32, "max": 35, "typical": 33.5}
    }
    
    return comfort_ranges.get(comfort_level, {"min": 20, "max": 25, "typical": 22.5}) 