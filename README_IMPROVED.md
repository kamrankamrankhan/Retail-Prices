# Retail Price Optimization Dashboard

A comprehensive Streamlit application for analyzing retail prices and training machine learning models to predict optimal pricing strategies.

## 🚀 Features

- **Interactive Dashboard**: Beautiful, responsive UI with real-time filtering
- **Advanced Visualizations**: Multiple chart types including histograms, scatter plots, correlation heatmaps
- **Machine Learning Models**: Decision Tree, Random Forest, and Gradient Boosting regressors
- **Data Validation**: Comprehensive data quality checks and validation
- **Performance Metrics**: Detailed model evaluation with multiple metrics
- **Time Series Analysis**: Price trends and seasonal patterns
- **Feature Importance**: Understanding which factors drive pricing

## 📊 Dataset

The application uses a retail price dataset containing:
- Product information (ID, category, name, description)
- Pricing data (unit price, total price, freight price)
- Competitor pricing (comp_1, comp_2, comp_3)
- Product metrics (score, weight, photos quantity)
- Temporal data (month, year, weekday, weekend, holiday)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Retail-Prices
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app_improved.py
   ```

## 📁 Project Structure

```
Retail-Prices/
├── app.py                    # Original Streamlit app
├── app_improved.py          # Enhanced version with better structure
├── Optimized.py             # Simplified version
├── data_validation.py       # Data validation utilities
├── requirements.txt         # Python dependencies
├── retail_price.csv         # Dataset
├── Retail Price Optimization.ipynb  # Jupyter notebook analysis
└── README.md               # This file
```

## 🔧 Configuration

### Environment Variables
- `STREAMLIT_SERVER_PORT`: Port for Streamlit server (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: localhost)

### Model Parameters
- **Test Size**: Percentage of data for testing (10-50%)
- **Random State**: Seed for reproducible results (1-100)
- **Model Types**: Decision Tree, Random Forest, Gradient Boosting

## 📈 Usage

### 1. Data Overview
- View dataset summary and statistics
- Check data quality score
- Identify missing values and outliers

### 2. Filtering
- Filter by price range
- Select specific product categories
- Adjust quantity ranges

### 3. Visualizations
- **Histogram**: Price distribution analysis
- **Box Plot**: Unit price outliers detection
- **Scatter Plot**: Quantity vs price relationships
- **Correlation Heatmap**: Feature relationships
- **Time Series**: Price trends over time
- **Bar Charts**: Category-wise analysis

### 4. Machine Learning
- Train multiple model types
- Compare performance metrics
- View feature importance
- Analyze prediction accuracy

## 🧪 Model Performance

The application provides comprehensive model evaluation:

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **Feature Importance**: Understanding key drivers

## 🔍 Data Validation

The application includes robust data validation:

- **Structure Validation**: Required columns and data types
- **Range Validation**: Reasonable value ranges
- **Missing Data Detection**: Comprehensive missing value analysis
- **Quality Scoring**: Overall data quality assessment

## 🚨 Error Handling

- File not found errors
- Data parsing errors
- Missing value handling
- Model training errors
- Visualization errors

## 📊 Performance Optimization

- **Caching**: Data loading and processing cached
- **Lazy Loading**: Components loaded on demand
- **Memory Management**: Efficient data handling
- **Responsive Design**: Optimized for different screen sizes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the code comments

## 🔄 Version History

- **v1.0**: Initial release with basic functionality
- **v2.0**: Enhanced UI and multiple model support
- **v2.1**: Added data validation and error handling
- **v2.2**: Performance optimizations and caching

## 🎯 Future Enhancements

- [ ] Real-time data integration
- [ ] Advanced forecasting models
- [ ] Export functionality for reports
- [ ] User authentication and sessions
- [ ] API endpoints for external integration
- [ ] Automated model retraining
- [ ] A/B testing framework
- [ ] Mobile-responsive design improvements