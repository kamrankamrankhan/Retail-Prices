#!/usr/bin/env python3
"""
Setup script for the Retail Price Optimization Dashboard.
This script helps users set up the project environment and run the application.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print welcome banner."""
    print("=" * 60)
    print("🚀 Retail Price Optimization Dashboard Setup")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible."""
    print("🔍 Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_pip():
    """Check if pip is available."""
    print("🔍 Checking pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available!")
        return False

def create_virtual_environment():
    """Create virtual environment."""
    print("🔧 Creating virtual environment...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("⚠️  Virtual environment already exists")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_activation_command():
    """Get the correct activation command based on OS."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    # Determine pip command based on OS
    if platform.system() == "Windows":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    try:
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_data_file():
    """Check if the data file exists."""
    print("🔍 Checking data file...")
    
    data_file = Path("retail_price.csv")
    if data_file.exists():
        print("✅ Data file found")
        return True
    else:
        print("❌ Data file 'retail_price.csv' not found!")
        print("   Please ensure the data file is in the project directory")
        return False

def run_tests():
    """Run unit tests."""
    print("🧪 Running tests...")
    
    # Determine python command based on OS
    if platform.system() == "Windows":
        python_cmd = "venv\\Scripts\\python"
    else:
        python_cmd = "venv/bin/python"
    
    try:
        result = subprocess.run([python_cmd, "test_app.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ All tests passed")
            return True
        else:
            print("⚠️  Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print()
    print("To run the application:")
    print()
    
    activation_cmd = get_activation_command()
    print(f"1. Activate virtual environment:")
    print(f"   {activation_cmd}")
    print()
    print("2. Run the application:")
    print("   streamlit run app_improved.py")
    print()
    print("3. Open your browser and go to:")
    print("   http://localhost:8501")
    print()
    print("📚 Available applications:")
    print("   - app_improved.py    (Enhanced version with all features)")
    print("   - app.py            (Original version)")
    print("   - Optimized.py      (Simplified version)")
    print()
    print("🔧 Additional commands:")
    print("   - Run tests: python test_app.py")
    print("   - Check data: python -c \"from data_validation import *; print('Data validation ready')\"")
    print()

def main():
    """Main setup function."""
    print_banner()
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # Setup environment
    if not create_virtual_environment():
        sys.exit(1)
    
    if not install_dependencies():
        sys.exit(1)
    
    # Check data file
    if not check_data_file():
        print("\n⚠️  Setup completed but data file is missing!")
        print("   The application will not work without the data file.")
        print("   Please add 'retail_price.csv' to the project directory.")
        print()
    
    # Run tests
    run_tests()
    
    # Print instructions
    print_usage_instructions()

if __name__ == "__main__":
    main()