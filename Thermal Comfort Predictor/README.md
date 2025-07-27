# Thermal Comfort Predictor

A machine learning system that predicts Thermal Sensation Vote (TSV) using environmental and personal parameters. The system employs a stacked ensemble approach combining multiple base models with a meta-learner for robust predictions.

## About

### Overall Flow
1. **Data Input**: Processes environment-specific data from Excel sheets (Classroom, Hostel, Workshop, etc.)
2. **Feature Engineering**:
   - Environment-specific feature selection
   - Missing value handling
   - Outlier capping using IQR method
   - Feature scaling using RobustScaler
   - Categorical encoding for personal factors

3. **Model Pipeline**:
   ```
   Raw Data → Preprocessing → Base Models → Meta Model → Final Predictions
                                ↓
                         Performance Metrics
                         Visualizations
                         Statistical Analysis
   ```

### Base Models
The system uses four diverse base models, each chosen for specific strengths:

1. **CatBoost Regressor**
   - Handles categorical features automatically
   - Robust to overfitting
   - Works well with numerical and categorical data

2. **Extra Trees Regressor**
   - Reduces variance through extreme randomization
   - Captures non-linear relationships
   - Good for feature importance analysis

3. **Elastic Net**
   - Combines L1 and L2 regularization
   - Handles multicollinearity
   - Provides stable linear predictions

4. **XGBoost Regressor**
   - High performance gradient boosting
   - Handles missing values well
   - Strong regularization capabilities

### Meta Model (LightGBM)
The meta-model combines predictions from base models because:
- Leverages strengths of each base model
- Reduces individual model biases
- Improves prediction stability
- Adapts to different environments

## Setup

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

3. **Data Preparation**
   - Place your input data in `dataset/input_dataset.xlsx`
   - Each environment should be in a separate sheet
   - Required columns:
     - Classroom: ['RATemp', 'MRT', 'Top', 'Air Velo', 'RH']
     - Other environments: Above + ['Clothing', 'Activity']
     - Target column: 'Given Final TSV'

## How to Run

1. **Basic Usage**
   ```bash
   python main.py
   ```

2. **Output**
   The system generates in the `output/` directory:
   - Predictions CSV for each environment
   - Statistical summaries
   - Feature importance plots
   - Actual vs Predicted plots
   - Error distribution analysis
   - Correlation heatmaps
   - Final results summary

## Results

The system evaluates predictions using multiple metrics:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² Score
- Custom Accuracy
- Residual Standard Deviation
- MBE (Mean Bias Error)

Current performance across environments:
```
Environment            RMSE    MAE     R²      Accuracy
Classroom             0.779   0.656   0.206   42.81%
Hostel                0.994   0.781   0.249   39.92%
Hostel_Winter         0.934   0.778   0.037   35.85%
Workshop/Laboratory   0.871   0.682   0.481   46.51%
```

## Explanation

### Model Performance
- Workshop/Laboratory environment shows the best performance (R² = 0.481)
- Winter conditions are more challenging to predict (R² = 0.037)
- Accuracy ranges from 35-47% across environments

### Feature Importance
- Environmental factors (temperature, humidity) generally show strong influence
- Personal factors (clothing, activity) provide additional context where available
- Feature importance varies by environment type

### Limitations
- Performance varies significantly between environments
- Winter conditions show lower prediction accuracy
- Model assumes consistent measurement conditions
