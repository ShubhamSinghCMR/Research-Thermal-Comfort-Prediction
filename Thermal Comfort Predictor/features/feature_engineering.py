import pandas as pd
import numpy as np

def compute_bmi(weight, height):
    try:
        return weight / (height ** 2)
    except:
        return np.nan

def collapse_perception(df, group, mapping):
    # Returns mapped value from binary one-hot perception columns
    def map_row(row):
        for col in group:
            if row.get(col, 0) == 1:
                return mapping.get(col.strip().lower(), 0)
        return 0
    return df.apply(map_row, axis=1)

def extract_perception_features(df):
    humidity_cols = [
        "Very Dry", "Moderately Dry", "Slightly dry", "Neutral",
        "Slightly Humid", "Moderately Humid", "Very Humid"
    ]
    air_cols = [
        "Very still", "Moderately still", "slightly still", "Acceptable",
        "Slightly Moving", "Moderately Moving", "Much Moving"
    ]
    light_cols = [
        "Very Bright", "Bright", "Slightly Bright", "Neither Bright nor Neither Dim",
        "Slightly Dim", "Dim", "Very Dim"
    ]

    humidity_map = {
        "very dry": -3, "moderately dry": -2, "slightly dry": -1,
        "neutral": 0, "slightly humid": 1, "moderately humid": 2, "very humid": 3
    }
    air_map = {
        "very still": -2, "moderately still": -1, "slightly still": 0,
        "acceptable": 0, "slightly moving": 1, "moderately moving": 2, "much moving": 3
    }
    brightness_map = {
        "very dim": -3, "dim": -2, "slightly dim": -1,
        "neither bright nor neither dim": 0,
        "slightly bright": 1, "bright": 2, "very bright": 3
    }

    df['HumidityPerception'] = collapse_perception(df, humidity_cols, humidity_map)
    df['AirMovement'] = collapse_perception(df, air_cols, air_map)
    df['LightPerception'] = collapse_perception(df, light_cols, brightness_map)

    return df

def calculate_clo_score(df, clothing_columns):
    clo_values = {
        'T-Shirt': 0.2,
        'Short sleeves shirt (Poly/cotton)': 0.25,
        'Long sleeves shirt (Poly/cotton)': 0.3,
        'Jacket/wwoolen jacket': 0.4,
        'Pullover/Sweater/upcoller': 0.3,
        'Thermal tops': 0.4,
        'Suit': 0.6,
        'Tights': 0.2,
        'Pyjamas': 0.3,
        'Lower (thermal inner)': 0.3,
        'Dhoti': 0.2,
        'Jeans': 0.35,
        'Trousers/long skirt (Poly/cotton)': 0.35,
        'Shorts/short skirt (Poly/cotton)': 0.2
    }

    # Convert clothing columns to numeric, handling any string values
    for col in clothing_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['CLO_score'] = df[clothing_columns].apply(
        lambda row: sum(row[col] * clo_values.get(col.strip(), 0) for col in clothing_columns),
        axis=1
    )

    return df

def calculate_met(df):
    MET_map = {
        'Sleeping hrs': 0.9,
        'Sitting (passive work) hrs': 1.0,
        'Sitting (Active work) hrs': 1.3,
        'Standing (relaxed )hrs': 1.5,
        'Standing (working)': 1.8,
        'Walking Indoors (hrs)': 2.0,
        'Walking (Outdoor) hrs': 2.5,
        'Others hrs': 1.2
    }

    # Convert MET columns to numeric, handling any string values
    met_columns = [col for col in MET_map.keys() if col in df.columns]
    for col in met_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['MET_score'] = df[met_columns].apply(
        lambda row: sum(row[k] * MET_map[k] for k in met_columns),
        axis=1
    )

    return df

def derive_tsv(df):
    vote_map = {
        'Cold': -2,
        'Cool': -1,
        'Slightly cool': -0.5,
        'Neutral': 0,
        'Slightly Warm': 0.5,
        'Warm': 1,
        'Hot': 2
    }

    def get_tsv(row):
        for col, score in vote_map.items():
            if str(row.get(col, 0)).strip() == '1':
                return score
        return np.nan

    df['TSV'] = df.apply(get_tsv, axis=1)
    return df

def engineer_features(df):
    df = df.copy()
    df.fillna(0, inplace=True)

    # Convert weight and height to numeric
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    df['Height'] = pd.to_numeric(df['Height'], errors='coerce')
    df['BMI'] = df.apply(lambda x: compute_bmi(x['weight'], x['Height']), axis=1)
    df = extract_perception_features(df)

    clothing_cols = [
        'T-Shirt', 'Short sleeves shirt (Poly/cotton)', 'Long sleeves shirt (Poly/cotton)',
        'Jacket/wwoolen jacket', 'Pullover/Sweater/upcoller', 'Thermal tops', 'Suit', 'Tights',
        'Pyjamas', 'Lower (thermal inner)', 'Dhoti', 'Jeans',
        'Trousers/long skirt (Poly/cotton)', 'Shorts/short skirt (Poly/cotton)'
    ]
    clothing_cols = [col for col in clothing_cols if col in df.columns]
    df = calculate_clo_score(df, clothing_cols)

    df = calculate_met(df)

    control_cols = [
        'Fan      O/C-        3',
        'Evaporative cooler O/C -                4         ',
        'Air conditioner       O/C -                5        ',
        'Window O/C -                1         ',
        'Door      O/C-        2'
    ]
    control_cols = [col for col in control_cols if col in df.columns]
    
    # Convert control columns to numeric, handling any string values
    for col in control_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['ThermalControlIndex'] = df[control_cols].sum(axis=1)

    df = derive_tsv(df)
    return df
