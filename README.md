# Pairs Trading Analysis System
A comprehensive Python-based system for analyzing and backtesting pairs trading strategies using statistical arbitrage techniques.

## Overview
This system provides tools for:
- Fetching and managing historical stock price data
- Analyzing potential trading pairs using statistical methods
- Training machine learning models to predict pair trading success
- Backtesting pairs trading strategies with configurable parameters
- Generating detailed performance analytics

## Project Structure
```
├── analysis.py          # Machine learning model for pair analysis
├── data_manager.py      # Data fetching and database management
├── main.py             # Main execution script
└── pairs_trading.py    # Core pairs trading logic and backtesting
```

## How to Run

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pairs-trading-system
```

2. Install required dependencies:
```bash
pip install pandas numpy yfinance scikit-learn statsmodels sqlite3
```

### Configuration

1. Edit `main.py` to configure your trading pairs and parameters:
```python
pairs = [
    ('AAPL', 'MSFT'),
    ('AAPL', 'NVDA'),
    ('GOOGL', 'META'),
    ('JPM', 'BAC')
]

start_date = '2022-01-01'
end_date = '2023-12-31'
```

2. Adjust trading parameters in `main.py` if desired:
```python
trader = PairsTrader(
    window=60,
    entry_threshold=1.5,
    exit_threshold=0.5,
    stop_loss=-0.03
)
```

### Running the System

1. Execute the main script:
```bash
python main.py
```

The system will:
- Fetch historical data for specified pairs
- Train the machine learning model
- Analyze each trading pair
- Run backtests where appropriate
- Generate performance metrics

### Output
The system will display:
- Data fetching progress
- Pair analysis results
- Trading signals
- Performance metrics including:
  - Total returns
  - Sharpe ratio
  - Monthly statistics
  - Maximum drawdown
  - Trade-specific statistics

## Components

### Data Manager (`data_manager.py`)
- Handles data fetching from Yahoo Finance
- Manages local SQLite database for storing historical price data
- Provides efficient data retrieval and storage mechanisms

### Pairs Analyzer (`analysis.py`)
- Implements machine learning-based pair analysis using Random Forest
- Features include correlation, volatility ratios, and spread volatility
- Predicts success probability for potential trading pairs

### Pairs Trader (`pairs_trading.py`)
- Implements core pairs trading strategy logic
- Features include:
  - Dynamic hedge ratio calculation
  - Z-score based entry/exit signals
  - Configurable stop-loss and position holding periods
  - Comprehensive performance analytics
- Provides detailed trade statistics and performance metrics

### Main Script (`main.py`)
- Orchestrates the entire trading system
- Configurable parameters for:
  - Trading pairs selection
  - Date ranges
  - Strategy parameters
- Generates comprehensive analysis reports

## Key Features
- **Data Management**
  - Automated data fetching from Yahoo Finance
  - Efficient local storage using SQLite
  - Handling of missing data and errors
- **Pair Analysis**
  - Statistical correlation analysis
  - Machine learning-based success prediction
  - Feature engineering for pair characteristics
- **Trading Strategy**
  - Dynamic hedge ratio calculation
  - Z-score based entry/exit signals
  - Stop-loss implementation
  - Maximum position holding periods
  - Flexible parameter configuration
- **Performance Analytics**
  - Total returns calculation
  - Sharpe ratio analysis
  - Monthly statistics
  - Maximum drawdown calculation
  - Detailed trade-by-trade analysis

## Configuration Parameters

### Pairs Trader
- `window`: Rolling window size for calculations (default: 60)
- `entry_threshold`: Z-score threshold for trade entry (default: 1.5)
- `exit_threshold`: Z-score threshold for trade exit (default: 0.5)
- `stop_loss`: Stop-loss threshold (default: -0.03)
- `max_position_hold`: Maximum days to hold position (default: 20)

### Pairs Analyzer
- `n_estimators`: Number of trees in Random Forest (default: 100)
- Features calculated:
  - Correlation between pairs
  - Volatility ratio
  - Spread volatility

## Performance Metrics
The system generates comprehensive performance metrics including:
- Total return
- Sharpe ratio
- Monthly statistics
  - Maximum monthly return
  - Minimum monthly return
  - Percentage of profitable months
- Maximum drawdown
- Trade-specific statistics
  - Win rate
  - Average trade return
  - Average winning/losing trade
