#!/bin/bash

# Retail Price Optimization Dashboard - Quick Start Script
# This script provides a quick way to set up and run the application

echo "🚀 Retail Price Optimization Dashboard - Quick Start"
echo "=================================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

echo "✅ Python and pip are available"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if data file exists
if [ ! -f "retail_price.csv" ]; then
    echo "⚠️  Warning: retail_price.csv not found!"
    echo "   The application will not work without the data file."
    echo "   Please ensure the data file is in the project directory."
    echo
fi

# Run tests
echo "🧪 Running tests..."
python test_app.py

echo
echo "🎉 Setup complete!"
echo
echo "To run the application:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run: streamlit run app_improved.py"
echo "3. Open browser: http://localhost:8501"
echo
echo "Available applications:"
echo "- app_improved.py (Enhanced version - Recommended)"
echo "- app.py (Original version)"
echo "- Optimized.py (Simplified version)"