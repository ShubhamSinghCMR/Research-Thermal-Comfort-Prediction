SEED = 42

CLO_MAPPING = {
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

MET_MAPPING = {
    'Sleeping hrs': 0.9,
    'Sitting (passive work) hrs': 1.0,
    'Sitting (Active work) hrs': 1.3,
    'Standing (relaxed )hrs': 1.5,
    'Standing (working)': 1.8,
    'Walking Indoors (hrs)': 2.0,
    'Walking (Outdoor) hrs': 2.5,
    'Others hrs': 1.2
}

HUMIDITY_SCALE = {
    "very dry": -3,
    "moderately dry": -2,
    "slightly dry": -1,
    "neutral": 0,
    "slightly humid": 1,
    "moderately humid": 2,
    "very humid": 3
}

AIR_SCALE = {
    "very still": -2,
    "moderately still": -1,
    "slightly still": 0,
    "acceptable": 0,
    "slightly moving": 1,
    "moderately moving": 2,
    "much moving": 3
}

LIGHT_SCALE = {
    "very dim": -3,
    "dim": -2,
    "slightly dim": -1,
    "neither bright nor neither dim": 0,
    "slightly bright": 1,
    "bright": 2,
    "very bright": 3
}

MODEL_DIR = "models/saved/"
TSV_COMFORT_RANGE = (-1, 1)
ENABLE_RULE_CORRECTION = True
TSV_CLIP_RANGE = (-3.0, 3.0)
