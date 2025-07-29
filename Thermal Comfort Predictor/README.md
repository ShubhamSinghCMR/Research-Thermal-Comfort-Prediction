# Thermal Comfort Predictor

A sophisticated machine learning system that predicts Thermal Sensation Vote (TSV) using environmental and personal parameters. The system implements a stacking ensemble approach combining multiple base models with a meta-learner for robust predictions across different environmental contexts.

### Model Stack

1. **Base Models Layer**
   - **CatBoost Regressor**
     - Depth: 7
     - Iterations: 1000
     - Learning rate: 0.05
     - Early stopping: 50 rounds
     - Strengths: Handles categorical features, robust to overfitting

   - **Extra Trees Regressor**
     - Trees: 500
     - Max depth: 18
     - Min samples split: 5
     - Min samples leaf: 3
     - Strengths: Reduces variance, captures non-linear relationships

   - **Elastic Net**
     - Alpha: 0.3
     - L1 ratio: 0.5
     - Max iterations: 5000
     - Strengths: Handles multicollinearity, stable linear predictions

   - **XGBoost Regressor**
     - Trees: 700
     - Max depth: 6
     - Learning rate: 0.05
     - Early stopping: 50 rounds
     - Strengths: High performance, handles missing values

2. **Meta-Model Layer (LightGBM)**
   - Environment-specific parameters
   - Early stopping with 50 rounds patience
   - Feature importance tracking
   - K-Fold cross-validation
   - Strengths:
     - Optimal combination of base predictions
     - Environment-specific adaptation
     - Model interpretability
     - Robust performance

## Technical Implementation

### Feature Engineering (`features/feature_engineering.py`)
1. **Data Cleaning**
   - Standardization of numeric inputs
   - Smart missing value handling
   - Outlier capping using IQR method (1.5 * IQR)

2. **Feature Processing**
   - Environment-specific feature selection
   - RobustScaler for numeric features
   - Categorical encoding for clothing and activity
   - Required Features:
     - Base: RATemp, MRT, Top, Air Velo, RH
     - Additional: Clothing, Activity (non-classroom)
     - Target: Given Final TSV

### Model Training (`models/`)
1. **Base Models** (`base_models.py`)
   - K-Fold cross-validation (5 folds)
   - Early stopping for boosting models
   - Out-of-fold predictions generation
   - Comprehensive metrics tracking

2. **Meta-Model** (`meta_model.py`)
   - LightGBM meta-learner
   - Environment-specific optimization
   - Feature importance analysis
   - Cross-validation performance tracking

## Setup and Usage

### Prerequisites
- Python 3.7+
- Virtual environment (recommended)
- Input data in Excel format

### Installation
1. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Data Requirements
- File: `dataset/input_dataset.xlsx`
- Sheet per environment
- Required columns:
  - Classroom: ['RATemp', 'MRT', 'Top', 'Air Velo', 'RH']
  - Other environments: Above + ['Clothing', 'Activity']
  - Target: 'Given Final TSV'

## Running the System

### 1. Main Pipeline
The main pipeline trains and evaluates the stacked ensemble model:

```bash
python run_main_pipeline.py
```

This will:
- Load and preprocess data from all environment sheets
- Train base models (CatBoost, ExtraTrees, ElasticNet, XGBoost)
- Generate out-of-fold predictions
- Train meta-model (LightGBM)
- Output predictions and performance metrics
- Generate visualization plots

Output files in `output/`:
- `predictions_{environment}.csv`: Predictions for each environment
- `feature_importance_{environment}.png`: Feature importance plots
- `actual_vs_predicted_{environment}.png`: Scatter plots
- `error_distribution_{environment}.png`: Error analysis
- `correlation_matrix_{environment}.png`: Feature correlations
- `summary_results.csv`: Overall performance metrics

### 2. Adaptive Pipeline
The adaptive pipeline implements environment-specific model adjustments:

```bash
python run_adaptive_pipeline.py
```

This performs:
- Environment-specific feature selection
- Adaptive hyperparameter tuning
- Custom model selection per environment
- Enhanced cross-validation strategy

Output in `adaptive_tsv_pipeline/output/`:
- `adaptive_predictions_{environment}.csv`: Environment-tuned predictions
- `adaptive_metrics_{environment}.json`: Detailed performance metrics
- `adaptive_model_params_{environment}.json`: Optimized parameters
- `adaptive_feature_analysis_{environment}.png`: Feature analysis plots

### 3. Compare Results
Compare predictions from both pipelines:

```bash
python compare_predicted_and_adaptive_results.py
```

This generates:
- `comparison_results.csv`: Side-by-side performance comparison
- Statistical significance tests
- Detailed error analysis
- Environment-specific insights

The comparison helps identify:
- Which pipeline performs better for each environment
- Where adaptive strategies provide significant improvements
- Potential areas for model enhancement
- Environment-specific optimization opportunities

## Performance Metrics

The system tracks multiple evaluation metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- Custom Accuracy within tolerance
- Residual Standard Deviation
- Train vs Validation accuracy gap

Current performance across environments:
```
Environment            RMSE    MAE     R²      Accuracy
Classroom             0.779   0.656   0.206   42.81%
Hostel                0.994   0.781   0.249   39.92%
Hostel_Winter         0.934   0.778   0.037   35.85%
Workshop/Laboratory   0.871   0.682   0.481   46.51%
```

## System Advantages

1. **Robustness**
   - Multiple diverse base models
   - Cross-validation at both layers
   - Outlier and missing data handling

2. **Adaptability**
   - Environment-specific modeling
   - Feature importance analysis
   - Automatic parameter tuning

3. **Quality Control**
   - Comprehensive metrics tracking
   - Overfitting detection
   - Performance validation

4. **Interpretability**
   - Feature importance analysis
   - Error distribution analysis
   - Model contribution tracking

## Limitations and Considerations

1. **Environmental Factors**
   - Performance varies by environment
   - Winter conditions more challenging
   - Assumes consistent measurements

2. **Data Quality**
   - Depends on input data quality
   - Sensitive to measurement errors
   - Requires complete feature sets

3. **Model Constraints**
   - Computational overhead of ensemble
   - Memory requirements for large datasets
   - Training time with cross-validation
