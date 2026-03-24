# 🚀 Retail Price Optimization Dashboard

![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A comprehensive, production-ready retail price optimization dashboard with machine learning models, price prediction, competitor analysis, and dynamic pricing strategies.

## ✨ Features

### 📊 **Dashboard Capabilities**
- **Interactive Visualizations**: Multiple chart types including histograms, box plots, scatter plots, correlation heatmaps
- **Real-time Price Prediction Calculator**: Input product parameters for instant predictions
- **Advanced Analytics Dashboard**: KPI metrics, market analysis, and data exploration tools
- **Export & Reporting**: Download filtered data in CSV, Excel, or JSON format

### 🤖 **Machine Learning Models**
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Linear/Ridge/Lasso Regression
- Model evaluation with multiple metrics (R², RMSE, MAE, MAPE)
- Cross-validation support
- Feature importance analysis

### 💰 **Price Optimization**
- **Multiple Pricing Strategies**:
  - Cost-plus pricing
  - Competitor-based pricing
  - Value-based pricing
  - Dynamic pricing
  - Penetration pricing
  - Price skimming
- Price elasticity analysis
- Competitive position analysis
- Profit margin optimization

### 📈 **Analytics Engine**
- Statistical analysis (descriptive stats, correlations, hypothesis testing)
- Trend analysis and forecasting
- Category performance metrics
- Customer behavior analysis
- Product segmentation

### 🔌 **API Endpoints**
- RESTful API with FastAPI
- Price prediction endpoint
- Optimization endpoints
- Batch processing support
- Health monitoring

## 🏗️ Project Structure

```
Retail-Prices/
├── app.py                    # Main Streamlit application
├── Optimized.py              # Enhanced Streamlit dashboard
├── config.py                 # Configuration settings
├── data_validation.py        # Data validation utilities
├── retail_price.csv          # Dataset
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose setup
│
├── models/                   # Machine Learning Models
│   ├── __init__.py
│   ├── price_predictor.py    # Price prediction models
│   ├── model_evaluator.py    # Model evaluation tools
│   └── feature_importance.py # Feature importance analysis
│
├── utils/                    # Utility Modules
│   ├── __init__.py
│   ├── price_optimizer.py    # Price optimization engine
│   ├── data_preprocessing.py # Data preprocessing utilities
│   ├── analytics.py          # Analytics engine
│   ├── visualizations.py     # Chart building utilities
│   ├── logger.py             # Logging configuration
│   ├── database.py           # Database connector
│   └── report_generator.py   # Report generation
│
├── api/                      # API Module
│   └── main.py               # FastAPI endpoints
│
├── pages/                    # Streamlit Pages
│   ├── 1_📦_Inventory_Analysis.py
│   ├── 2_📈_Sales_Forecast.py
│   └── 3_🏪_Competitor_Analysis.py
│
├── tests/                    # Test Suite
│   ├── conftest.py           # Pytest configuration
│   └── test_modules.py       # Unit tests
│
└── .github/                  # GitHub Actions
    └── workflows/
        └── ci.yml            # CI/CD pipeline
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/kamrankamrankhan/Retail-Prices.git
cd Retail-Prices
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**
```bash
streamlit run app.py
```

5. **Run the FastAPI server (optional)**
```bash
uvicorn api.main:app --reload --port 8000
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access the dashboard
# Streamlit: http://localhost:8501
# API: http://localhost:8000
```

## 📊 Usage

### Price Prediction

```python
from models.price_predictor import PricePredictor
import pandas as pd

# Load data
data = pd.read_csv('retail_price.csv')
data['comp_price_diff'] = data['unit_price'] - data['comp_1']

# Initialize and train model
predictor = PricePredictor(model_type='random_forest')
predictor.fit(data)

# Make prediction
predicted_price = predictor.predict_single(
    qty=100,
    unit_price=50.0,
    comp_price=45.0,
    product_score=4.5
)
print(f"Predicted Price: ${predicted_price:.2f}")
```

### Price Optimization

```python
from utils.price_optimizer import PriceOptimizer, PricingStrategy
import pandas as pd

data = pd.read_csv('retail_price.csv')
optimizer = PriceOptimizer(data)

# Optimize price using competitor-based strategy
result = optimizer.optimize_price(
    product_id='bed1',
    strategy=PricingStrategy.COMPETITOR_BASED,
    position='competitive'
)

print(f"Optimal Price: ${result.optimal_price}")
print(f"Expected Revenue: ${result.expected_revenue}")
```

### Data Preprocessing

```python
from utils.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(data)
preprocessor.remove_duplicates()\
    .handle_missing_values(strategy='mean')\
    .remove_outliers(method='iqr')\
    .create_features()

processed_data = preprocessor.get_processed_data()
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict` | POST | Price prediction |
| `/optimize` | POST | Price optimization |
| `/optimize/batch` | POST | Batch optimization |
| `/products` | GET | List products |
| `/categories` | GET | List categories |
| `/analytics/summary` | GET | Analytics summary |

### Example API Request

```bash
# Predict price
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "qty": 100,
    "unit_price": 50.0,
    "comp_1": 45.0,
    "product_score": 4.5
  }'
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test file
python tests/test_modules.py
```

## 📈 Dashboard Pages

1. **Main Dashboard** (`app.py`) - Core analytics and visualizations
2. **Inventory Analysis** - Stock levels and product performance
3. **Sales Forecast** - Predictive sales analytics
4. **Competitor Analysis** - Market position comparison

## 🛠️ Configuration

Edit `config.py` to customize:

```python
# Application settings
APP_TITLE = "Retail Price Analytics Dashboard"
DATA_FILE = "retail_price.csv"

# Model settings
DEFAULT_TEST_SIZE = 20
DEFAULT_RANDOM_STATE = 42

# Data validation thresholds
MAX_PRICE_THRESHOLD = 10000
MIN_QUALITY_SCORE = 70
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Streamlit for the amazing dashboard framework
- Scikit-learn for machine learning utilities
- Plotly for interactive visualizations
- FastAPI for the high-performance API framework

## 📞 Support

For issues and feature requests, please use the [GitHub Issues](https://github.com/kamrankamrankhan/Retail-Prices/issues) page.

---

**Built with ❤️ for Retail Analytics**
