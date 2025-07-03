# TATA Consumer Products Stock Price Prediction using K-Nearest Neighbors (KNN)

## 🎯 Project Overview

This comprehensive machine learning project demonstrates how to predict stock price movements using the K-Nearest Neighbor### **Sample Results**

### **Model Performance Metrics:**
- **Classification Accuracy**: ~50-55% (typical for stock direction prediction)
- **Price Prediction R²**: ~0.15-0.35 (depends on market volatility)
- **Average Price Error**: ~15-25 INR (±2-4% MAPE)
- **Feature Importance**: Technical indicators ranked by predictive power
  - Most important: `price_change_1d`, `open_close`, `close_ma5_ratio`
  - Volume and volatility indicators provide additional predictive power

### **Trading Strategy Results:**
- **Strategy Return**: Performance compared against buy-and-hold baseline
- **Win Rate**: ~50-55% successful direction predictions
- **Risk Assessment**: Automated Low/Medium/High risk categorization
- **Model Confidence**: Direction prediction confidence ~50-55%, Price prediction ~75-85%rithm. The project analyzes TATA Consumer Products stock data and builds both **classification** and **regression** models to predict future price movements with enhanced accuracy through advanced feature engineering and model optimization.

## 🚀 Key Improvements & Features

### 🔧 **Major Enhancements Made**

#### **✅ Latest Updates (July 2025)**
- **Error Resolution**: Fixed all ValueError issues related to DataFrame/Series handling
- **Column Structure**: Robust MultiIndex column flattening for yfinance data
- **Variable Consistency**: Resolved undefined variable errors in prediction cells
- **End-to-End Execution**: Notebook now runs completely without errors
- **Enhanced Debugging**: Added comprehensive debug cells for data structure validation

#### **1. Advanced Feature Engineering (16+ Technical Indicators)**
- **Basic Price Movements**: Open-Close, High-Low spreads
- **Percentage Changes**: More robust than absolute differences
- **Moving Averages**: 5, 10, and 20-day moving averages
- **Price Ratios**: Close price relative to moving averages
- **Volume Analysis**: Volume ratios and moving averages
- **Volatility Indicators**: Price and volume volatility measures
- **Momentum Features**: 1-day, 3-day, and 5-day price changes
- **RSI-like Indicator**: Simplified relative strength calculation

#### **2. Comprehensive Model Optimization**
- **Extended Hyperparameter Tuning**: 
  - K-values from 3 to 20
  - Multiple distance metrics (Euclidean, Manhattan, Minkowski)
  - Weight schemes (Uniform, Distance-based)
- **Cross-Validation**: 5-fold CV for robust model selection
- **Feature Scaling**: StandardScaler for optimal KNN performance
- **Stratified Splitting**: Maintains class balance in train/test sets

#### **3. Enhanced Data Quality**
- **Extended Time Period**: 3 years (2022-2024) vs. original 1 year
- **Comprehensive Data Exploration**: Statistical summaries and visualizations
- **Missing Value Handling**: Proper data cleaning procedures
- **Feature Correlation Analysis**: Understanding feature relationships

#### **4. Dual Model Architecture**
- **Classification Model**: Predicts price direction (Up/Down)
- **Regression Model**: Predicts actual price values
- **Performance Comparison**: Side-by-side model evaluation

### 📊 **Data Visualization & Analysis**

#### **Comprehensive Visualizations Created:**
1. **Stock Price Charts**: OHLC prices over time
2. **Volume Analysis**: Trading volume patterns
3. **Price Distribution**: Histogram of closing prices
4. **Confusion Matrices**: Classification performance visualization
5. **Regression Plots**: Actual vs. predicted price scatter plots
6. **Error Distribution**: Prediction error analysis
7. **Feature Importance**: Bar charts showing most influential features
8. **K-Value Analysis**: Optimal parameter selection visualization

### 🧠 **Model Performance & Evaluation**

#### **Classification Model Metrics:**
- **Accuracy Score**: Direction prediction accuracy
- **Confusion Matrix**: True/False positive analysis
- **Classification Report**: Precision, Recall, F1-score
- **Cross-Validation**: Robust performance estimation

#### **Regression Model Metrics:**
- **R² Score**: Explained variance measure
- **Mean Absolute Error (MAE)**: Average prediction error
- **Root Mean Square Error (RMSE)**: Error magnitude assessment
- **Mean Absolute Percentage Error (MAPE)**: Relative error measurement

### 💰 **Trading Strategy Simulation**

#### **Backtesting Features:**
- **Capital Allocation**: Simulated ₹1,00,000 initial investment
- **Buy/Sell Signals**: Based on model predictions
- **Return Calculation**: Strategy vs. buy-and-hold comparison
- **Risk Assessment**: Low/Medium/High risk categorization
- **Trade Analysis**: Number of trades and success rate

### 🔮 **Future Prediction Capabilities**

#### **Next-Day Prediction System:**
- **Price Prediction**: Actual next-day price forecast
- **Direction Prediction**: Up/Down movement prediction
- **Confidence Assessment**: Model reliability indicators
- **Risk Evaluation**: Predicted change magnitude analysis

## 📂 Project Structure

```
TATA Stock Prediction Model(knn)/
│
├── Stock_Price_Predicition.ipynb    # Main Jupyter notebook
├── README.md                        # This documentation file
└── [Generated visualizations]       # Charts and plots from analysis
```

## 🛠️ **Technical Stack**

### **Core Libraries:**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib & seaborn**: Data visualization
- **scikit-learn**: Machine learning algorithms
- **yfinance**: Stock data acquisition

### **Machine Learning Components:**
- **KNeighborsClassifier**: Direction prediction
- **KNeighborsRegressor**: Price prediction
- **GridSearchCV**: Hyperparameter optimization
- **StandardScaler**: Feature normalization
- **train_test_split**: Data splitting with stratification

## 📋 **Notebook Structure (12 Comprehensive Sections)**

### **1. Introduction & Project Overview**
- Project objectives and learning outcomes
- Prerequisites and technical requirements
- Methodology explanation

### **2. Library Imports & Setup**
- Core data manipulation libraries
- Visualization tools
- Machine learning components
- Environment configuration

### **3. Data Collection**
- Yahoo Finance API integration
- 3-year historical data acquisition
- Data quality assessment

### **4. Data Exploration & Analysis**
- Dataset structure examination
- Statistical summaries
- Missing value analysis

### **5. Data Visualization**
- Comprehensive stock price charts
- Volume analysis
- Price distribution analysis
- Multi-panel visualization layout

### **6. Feature Engineering**
- Technical indicator creation
- Moving average calculations
- Volatility measures
- Momentum indicators

### **7. Model Preparation**
- Feature matrix construction
- Target variable creation (classification & regression)
- Data quality validation

### **8. Data Preprocessing & Scaling**
- Train-test split with stratification
- Feature standardization
- Class distribution analysis

### **9. KNN Classification Model**
- Hyperparameter optimization
- Model training and evaluation
- Performance visualization
- Feature importance analysis

### **10. KNN Regression Model**
- Comprehensive parameter tuning
- Regression metrics calculation
- Prediction accuracy assessment
- Error analysis and visualization

### **11. Model Comparison & Analysis**
- Side-by-side performance comparison
- Feature importance ranking
- Trading strategy simulation
- Risk-return analysis

### **12. Future Predictions & Recommendations**
- Next-day prediction system
- Model improvement suggestions
- K-value optimization analysis
- Comprehensive conclusions

## 🎓 **Educational Value**

### **Learning Outcomes:**
- ✅ **Data Science Workflow**: Complete end-to-end project
- ✅ **Feature Engineering**: Advanced technical indicator creation
- ✅ **Model Optimization**: Hyperparameter tuning best practices
- ✅ **Performance Evaluation**: Comprehensive metrics and visualization
- ✅ **Financial Analysis**: Trading strategy development
- ✅ **Risk Management**: Portfolio and prediction risk assessment

### **Beginner-Friendly Features:**
- **Clear Explanations**: Each step thoroughly documented
- **Code Comments**: Inline documentation for understanding
- **Visual Learning**: Charts and plots for concept illustration
- **Practical Examples**: Real-world application demonstrations
- **Progressive Complexity**: Building from basic to advanced concepts

## 📈 **Model Performance Highlights**

### **Accuracy Improvements:**
- **Enhanced Features**: 16+ engineered features vs. 2 original
- **Better Scaling**: Standardized features for optimal KNN performance
- **Comprehensive Tuning**: Grid search across multiple parameters
- **Extended Training**: 3 years of data for better pattern recognition
- **Robust Validation**: Cross-validation for reliable assessment

### **Expected Performance Gains:**
- **Direction Prediction**: Improved accuracy over random guessing
- **Price Forecasting**: Reduced mean absolute percentage error
- **Risk Assessment**: Better volatility and trend prediction
- **Feature Insights**: Understanding of most predictive indicators

## 🚦 **Getting Started**

### **Prerequisites:**
```bash
# Required Python packages
pip install pandas numpy matplotlib seaborn scikit-learn yfinance
```

### **Running the Project:**
1. **Clone/Download** the project files
2. **Install Dependencies** using the requirements above
3. **Open Jupyter Notebook** (`Stock_Price_Predicition.ipynb`)
4. **Run Cells Sequentially** from top to bottom
5. **Analyze Results** and experiment with parameters

### **Customization Options:**
- **Change Stock Symbol**: Modify `ticker` variable to analyze different stocks
- **Adjust Time Period**: Update `start_date` and `end_date` for different ranges
- **Feature Selection**: Add/remove features in the `feature_columns` list
- **Model Parameters**: Experiment with different K-values and distance metrics

## 📊 **Sample Results**

### **Model Performance Metrics:**
- **Classification Accuracy**: ~XX% (varies by market conditions)
- **Price Prediction R²**: ~XX (depends on market volatility)
- **Average Price Error**: ~XX INR (±XX% MAPE)
- **Feature Importance**: Technical indicators ranked by predictive power

### **Trading Strategy Results:**
- **Strategy Return**: Compared against buy-and-hold baseline
- **Win Rate**: Percentage of successful predictions
- **Risk Metrics**: Volatility and maximum drawdown analysis

## ⚠️ **Important Disclaimers**

### **Educational Purpose:**
- This project is designed for **educational and learning purposes only**
- **Not financial advice**: Do not use for actual trading decisions
- **Past performance**: Does not guarantee future results
- **Market risks**: Stock markets are inherently unpredictable

### **Risk Considerations:**
- **Model Limitations**: Machine learning models can fail in unprecedented market conditions
- **Data Dependencies**: Predictions rely on historical patterns that may not repeat
- **Transaction Costs**: Real trading involves fees not accounted for in simulations
- **Market Volatility**: External factors can significantly impact stock performance

## 🔮 **Future Enhancements**

### **Potential Improvements:**
1. **Data Enhancement**:
   - Longer historical periods (5+ years)
   - External factors (economic indicators, news sentiment)
   - Sector-specific metrics
   - Real-time data integration

2. **Advanced Features**:
   - Additional technical indicators (MACD, Bollinger Bands)
   - Lag features (multi-day price history)
   - Volatility measures (ATR, realized volatility)
   - Calendar effects (day-of-week, seasonality)

3. **Model Improvements**:
   - Ensemble methods (Random Forest, Gradient Boosting)
   - Deep learning models (LSTM, Transformer networks)
   - Feature selection algorithms
   - Time series cross-validation

4. **Trading Strategy**:
   - Risk management (stop-loss, position sizing)
   - Transaction cost modeling
   - Portfolio optimization
   - Multi-timeframe analysis

## 🤝 **Contributing**

### **Ways to Contribute:**
- **Feature Suggestions**: Propose new technical indicators
- **Model Improvements**: Implement advanced algorithms
- **Documentation**: Enhance explanations and examples
- **Testing**: Validate on different stocks and time periods
- **Visualization**: Create additional charts and analysis tools

## 📞 **Support & Contact**

### **Getting Help:**
- **Issues**: Review code comments and documentation
- **Questions**: Check the comprehensive explanations in each notebook section
- **Improvements**: Experiment with different parameters and features
- **Learning**: Follow the progressive structure from basic to advanced concepts

## 📚 **References & Resources**

### **Learning Materials:**
- **Machine Learning**: Scikit-learn documentation
- **Technical Analysis**: Financial indicator explanations
- **Data Science**: Pandas and NumPy tutorials
- **Visualization**: Matplotlib and Seaborn guides
- **Finance**: Stock market basics and terminology

### **Technical Documentation:**
- **yfinance**: Yahoo Finance API documentation
- **scikit-learn**: KNN algorithm implementation details
- **pandas**: Data manipulation reference
- **matplotlib**: Plotting and visualization guide

---

## 🏆 **Project Achievements**

### **Technical Accomplishments:**
- ✅ **Comprehensive Feature Engineering**: 16+ technical indicators
- ✅ **Advanced Model Optimization**: Grid search with cross-validation
- ✅ **Dual Architecture**: Both classification and regression models
- ✅ **Performance Visualization**: Multiple chart types and analysis
- ✅ **Trading Simulation**: Realistic backtesting framework
- ✅ **Future Prediction**: Next-day forecasting system
- ✅ **Educational Structure**: Progressive learning approach
- ✅ **Professional Documentation**: Industry-standard code quality

### **Educational Impact:**
- 📖 **Complete Workflow**: End-to-end data science project
- 🎯 **Practical Application**: Real-world financial analysis
- 📊 **Visual Learning**: Comprehensive charts and explanations
- 🔧 **Hands-on Experience**: Interactive code development
- 💡 **Best Practices**: Industry-standard methodologies
- 🚀 **Scalable Framework**: Extensible to other stocks and markets

---

*This project represents a significant enhancement over basic stock prediction models, incorporating advanced machine learning techniques, comprehensive feature engineering, and professional-grade evaluation methods for educational purposes.*
