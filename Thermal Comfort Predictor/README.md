# Thermal Comfort Predictor

A comprehensive machine learning system for predicting thermal comfort and environmental temperature using advanced ensemble methods and meta-learning approaches.

## Project Overview

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
- **Automated Visualizations**: Generates 8+ professional-quality performance graphs automatically
- **Configurable Parameters**: Complete control over all model parameters and train/test splits
- **High Prediction Coverage**: Optimized architecture for maximum prediction output (131 predictions from 949 records)

## Performance Highlights

- **TSV Prediction**: R² = 0.852 (excellent correlation)
- **Temperature Prediction**: R² = 0.897 (outstanding accuracy)
- **Uncertainty Coverage**: 98.46% (reliable prediction intervals)
- **Prediction Coverage**: 131 out of 949 records (optimized for maximum output)

## Project Structure

```
Thermal Comfort Predictor/
├── dataset/
│   └── input_dataset.csv              # Main input dataset (thermal comfort survey data)
├── features/
│   └── feature_engineering.py         # Feature engineering pipeline
├── models/
│   ├── base_models.py                 # Base model training (CatBoost, RF, Bayesian Ridge, TabNet)
│   ├── meta_model.py                  # Meta-learning implementation (LightGBM)
│   ├── rule_correction.py             # Rule-based corrections for TSV
│   └── saved/                         # Trained model files and feature importance
├── utils/
│   ├── config.py                      # Configuration settings and model parameters
│   ├── metrics.py                     # Evaluation metrics and scoring functions
│   ├── thermal_comfort.py             # TSV-to-temperature conversion utilities
│   └── visualizations.py              # Automatic graph generation (8 professional plots)
├── output/                            # Generated output files
│   ├── complete_dataset_with_predictions.csv  # Full dataset with predictions
│   ├── results.csv                    # Clean results (predicted rows only)
│   ├── metrics_report.txt             # Performance metrics and statistics
│   └── graphs/                        # Automatically generated visualization plots
├── main.py                            # Main execution pipeline
├── requirements.txt                   # Python dependencies
└── README.md                          # This documentation
```

## Pipeline Flow

### 1. **Data Loading & Preprocessing**
- Loads thermal comfort survey data from `dataset/input_dataset.csv`
- Performs initial data cleaning and validation
- Handles missing values and data type conversions

### 2. **Feature Engineering**
- Computes derived features:
  - **BMI**: Body Mass Index from height and weight
  - **CLO Score**: Clothing insulation level based on garment combinations
  - **MET Score**: Metabolic activity level from activity patterns
  - **Perception Scores**: Thermal, air quality, and lighting perceptions
  - **Environmental Factors**: Humidity, air velocity, and comfort indices

### 3. **Model Training Pipeline**
- **Base Models**: Train CatBoost, Random Forest, Bayesian Ridge, and TabNet independently
- **Meta-Model**: LightGBM combines base model predictions with original features
- **Optimized Architecture**: Single train/test split for maximum prediction coverage
- **Rule Correction**: Applies empirical rules to improve TSV prediction realism

### 4. **Temperature Estimation**
- Converts TSV predictions to estimated temperature (°C)
- Uses thermal comfort models considering clothing, activity, humidity, air movement
- Applies domain-specific thermal comfort science principles

### 5. **Evaluation & Visualization**
- Calculates comprehensive performance metrics (R², RMSE, MAE, classification accuracy)
- Generates multiple output formats for different use cases
- **Automatically creates 8 professional visualization graphs**:
  - Performance summary dashboard with key metrics
  - Prediction scatter plots (TSV and Temperature vs. actual values)
  - 4-panel residual analysis for model diagnostics
  - Confusion matrix for comfort classification accuracy
  - Distribution comparisons and box plots
  - Uncertainty analysis with prediction intervals
  - Comfort level distribution analysis

## Quick Start

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
- Load and process your thermal comfort survey data (949 records)
- Train ensemble models and meta-learners with optimized architecture
- Generate predictions for TSV and temperature (131 predictions)
- Save results to the `output/` directory
- Generate 8 professional visualization graphs automatically
- Display comprehensive performance metrics in the console

## Output Files

The pipeline generates the following output files in the `output/` directory:

### **Data Files**
- **`complete_dataset_with_predictions.csv`**: Original input dataset with three new prediction columns:
  - `Predicted TSV Value`: Numeric TSV predictions (-2 to +2 scale)
  - `Predicted comfort`: Human-readable comfort levels (Very Cold, Cold, Cool, Comfortable, Warm, Hot, Very Hot)
  - `Predicted Temperature (degrees Celsius)`: Estimated temperature in degrees Celsius
- **`results.csv`**: Clean results file containing only rows with predictions and essential columns:
  - Survey metadata (Sr .No., Surveyr, Building Name, Location, Condition Type, Occupant Name, Occupant Designation)
  - Core predictions (Predicted TSV Value, Predicted comfort, Predicted Temperature (degrees Celsius))
- **`metrics_report.txt`**: Detailed performance metrics including R², RMSE, MAE, classification accuracy, and uncertainty statistics

### **Visualization Files (output/graphs/)**
The pipeline automatically generates 8 professional-quality visualization graphs:

1. **`performance_summary.png`**: Comprehensive dashboard showing R², RMSE, MAE for both TSV and temperature
2. **`tsv_predictions.png`**: Scatter plot of predicted vs. actual TSV values with trend line
3. **`temperature_predictions.png`**: Scatter plot of predicted vs. actual temperature with trend line
4. **`residual_analysis.png`**: 4-panel residual analysis for model diagnostics and bias detection
5. **`confusion_matrix.png`**: Classification accuracy matrix for comfort levels
6. **`distribution_comparison.png`**: Box plots comparing predicted vs. actual value distributions
7. **`uncertainty_analysis.png`**: Prediction intervals and uncertainty quantification visualization
8. **`comfort_levels.png`**: Distribution analysis of comfort level predictions

All graphs are saved at 300 DPI resolution for publication-ready quality.

## Configuration

The system is highly configurable through `utils/config.py`:

### **Train/Test Split**
- Adjustable percentage split (default: 80% train, 20% test)
- Validation ensures percentages sum to 100%
- Recommended configurations provided with explanations

### **Model Parameters**
Complete control over all model hyperparameters:
- **CatBoost**: Iterations, learning rate, depth, regularization
- **Random Forest**: Trees, depth, sampling parameters
- **Bayesian Ridge**: Prior distributions and regularization
- **LightGBM Meta-Model**: Boosting parameters and regularization
- **Quantile Random Forest**: Uncertainty quantification parameters

### **Performance Tuning Guidelines**
- Detailed instructions for handling overfitting/underfitting
- Speed vs. accuracy trade-off recommendations
- Parameter adjustment guidelines based on dataset characteristics

### **Reproducibility**
- Fixed SEED (42) ensures consistent results across runs
- All random states synchronized for reproducible experiments

## Performance Optimization

- **For Speed**: Reduce `n_estimators`, `iterations` in config
- **For Accuracy**: Increase `n_estimators`, reduce `learning_rate`
- **For Overfitting**: Increase regularization parameters, reduce model complexity
- **For Underfitting**: Decrease regularization, increase model complexity

## Future Enhancements

- Integration with real-time environmental sensors
- Web-based interface for interactive predictions
- Additional base models and ensemble techniques
- Cross-validation and hyperparameter optimization
- Support for different thermal comfort standards (ASHRAE, EN, ISO)
