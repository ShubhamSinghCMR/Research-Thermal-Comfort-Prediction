# Thermal Comfort Predictor

A comprehensive machine learning system for predicting thermal comfort and environmental temperature using advanced ensemble methods and meta-learning approaches.

## 🎯 Project Overview

This project implements a sophisticated thermal comfort prediction system that combines multiple machine learning models with domain-specific knowledge to accurately predict:

- **TSV (Thermal Sensation Vote)**: Subjective thermal comfort ratings on a scale from -2 (Cold) to +2 (Hot)
- **Temperature (°C)**: Estimated environmental temperature based on comfort responses and contextual factors
- **Comfort Classification**: Human-readable comfort levels (Very Cold, Cold, Cool, Comfortable, Warm, Hot, Very Hot)

### Key Features

- **Multi-Model Ensemble**: Combines predictions from CatBoost, Random Forest, Bayesian Ridge, and TabNet
- **Meta-Learning**: Uses base model predictions as features for final predictions via LightGBM
- **Rule-Based Correction**: Applies empirical rules to improve prediction realism
- **Uncertainty Quantification**: Provides prediction intervals using Quantile Random Forest
- **Temperature Estimation**: Converts comfort predictions to actual temperature estimates using thermal comfort science
- **Comprehensive Evaluation**: Multiple regression and classification metrics

## 📁 Project Structure

```
Thermal Comfort Predictor/
├── dataset/
│   └── temperature_dataset.csv        # Main input dataset (thermal comfort survey)
├── features/
│   └── feature_engineering.py         # Feature engineering pipeline
├── models/
│   ├── base_models.py                 # Base model training (CatBoost, RF, Bayesian Ridge, TabNet)
│   ├── meta_model.py                  # Meta-learning implementation (LightGBM)
│   ├── rule_correction.py             # Rule-based corrections for TSV
│   └── saved/                         # Trained model files and feature importance
├── utils/
│   ├── config.py                      # Configuration settings and constants
│   ├── metrics.py                     # Evaluation metrics and scoring functions
│   └── thermal_comfort.py             # TSV-to-temperature conversion utilities
├── output/                            # Generated output files
│   ├── complete_dataset_with_predictions.csv  # Full dataset with predictions
│   ├── results.csv                    # Clean results (predicted rows only)
│   └── metrics_report.txt             # Performance metrics and statistics
├── main.py                            # Main execution pipeline
├── requirements.txt                   # Python dependencies
└── README.md                          # This documentation
```

## 🔄 Pipeline Flow

### 1. **Data Loading & Preprocessing**
- Loads thermal comfort survey data from `dataset/temperature_dataset.csv`
- Performs initial data cleaning and validation

### 2. **Feature Engineering**
- Computes derived features:
  - **BMI**: Body Mass Index from height and weight
  - **CLO Score**: Clothing insulation level
  - **MET Score**: Metabolic activity level
  - **Perception Scores**: Thermal, air quality, and lighting perceptions
  - **Environmental Factors**: Humidity, air velocity, etc.

### 3. **Model Training Pipeline**
- **Base Models**: Train CatBoost, Random Forest, Bayesian Ridge, and TabNet
- **Meta-Model**: LightGBM combines base model predictions with context features
- **Rule Correction**: Applies empirical rules to improve TSV predictions

### 4. **Temperature Estimation**
- Converts TSV predictions to estimated temperature (°C)
- Uses thermal comfort models considering clothing, activity, humidity, air movement

### 5. **Evaluation & Output**
- Calculates comprehensive performance metrics
- Generates multiple output formats for different use cases

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Thermal-Comfort-Predictor
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate (Windows)
   venv\Scripts\activate
   
   # Activate (macOS/Linux)
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

Execute the complete thermal comfort prediction pipeline:

```bash
python main.py
```

The pipeline will:
- Load and process your thermal comfort survey data
- Train ensemble models and meta-learners
- Generate predictions for TSV and temperature
- Save results to the `output/` directory
- Display performance metrics in the console

## 📊 Output Files

The pipeline generates three main output files:

### 1. **`complete_dataset_with_predictions.csv`**
- **Purpose**: Complete dataset with predictions in context
- **Content**: All original survey data + three new prediction columns
- **Use Case**: Full analysis and debugging

### 2. **`results.csv`**
- **Purpose**: Clean, focused results for reporting
- **Content**: Only rows with predictions + key survey information
- **Columns**: Survey details + Predicted TSV Value, Predicted comfort, Predicted Temperature (degrees Celsius)
- **Use Case**: Business reporting and stakeholder presentations

### 3. **`metrics_report.txt`**
- **Purpose**: Detailed performance analysis
- **Content**: Regression metrics, classification scores, uncertainty coverage
- **Use Case**: Model evaluation and research analysis

## 🧠 Model Architecture

### Base Models
- **CatBoost**: Gradient boosting with categorical feature handling
- **Random Forest**: Ensemble of decision trees with uncertainty quantification
- **Bayesian Ridge**: Linear regression with Bayesian regularization
- **TabNet**: Deep learning for tabular data (fallback if available)

### Meta-Learning Strategy
- **Input**: Base model predictions + original features
- **Algorithm**: LightGBM gradient boosting
- **Output**: Final TSV and temperature predictions
- **Benefits**: Reduces overfitting, improves generalization

### Rule-Based Corrections
- **Purpose**: Ensure predictions align with thermal comfort science
- **Method**: Empirical rules based on dataset analysis
- **Impact**: Improves prediction realism and interpretability

### Temperature Estimation
- **Method**: Thermal comfort models (PMV/PPD approach)
- **Factors**: Clothing insulation, activity level, humidity, air velocity
- **Output**: Estimated temperature in degrees Celsius

## 📈 Performance Metrics

The system evaluates performance across multiple dimensions:

### Regression Metrics
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error  
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination

### Classification Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Uncertainty Metrics
- **Quantile Coverage**: Percentage of true values within prediction intervals
- **Prediction Intervals**: Lower and upper bounds for uncertainty quantification

## ⚙️ Configuration

Key parameters can be modified in `utils/config.py`:

```python
# Model settings
SEED = 42                    # Random seed for reproducibility
MODEL_DIR = "models/saved/"  # Directory for saved models

# Comfort thresholds
TSV_COMFORT_RANGE = {
    'very_cold': -2.5,
    'cold': -1.5,
    'cool': -0.5,
    'comfortable': 0.5,
    'warm': 1.5,
    'hot': 2.5
}
```

## 🔧 Customization

### Adding New Features
1. Modify `features/feature_engineering.py`
2. Add feature computation functions
3. Update feature selection in `main.py`

### Modifying Models
1. Edit model parameters in respective files
2. Add new models to `models/base_models.py`
3. Update meta-feature preparation in `models/meta_model.py`

### Adjusting Rules
1. Modify rule logic in `models/rule_correction.py`
2. Update thresholds based on domain knowledge
3. Test with your specific dataset

## 📝 Logging

The system provides comprehensive logging:
- **Progress tracking**: Step-by-step pipeline execution
- **Performance metrics**: Model training and evaluation results
- **Error handling**: Detailed error messages and debugging info
- **File operations**: Confirmation of saved outputs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Thermal comfort research community
- ASHRAE standards for thermal comfort
- Open-source machine learning libraries
