from data_manager import DataManager
from pairs_trading import PairsTrader
from analysis import PairsAnalyzer
import pandas as pd
import numpy as np

def main():
    try:
        print("Starting pairs trading analysis...")
        
        data_manager = DataManager()
        trader = PairsTrader(
            window=60,
            entry_threshold=1.5,
            exit_threshold=0.5,
            stop_loss=-0.03
        )
        analyzer = PairsAnalyzer()

        pairs = [
            ('AAPL', 'MSFT'),
            ('AAPL', 'NVDA'),
            ('GOOGL', 'META'),
            ('JPM', 'BAC')
        ]

        start_date = '2022-01-01'
        end_date = '2023-12-31'
        
        symbols = list(set([stock for pair in pairs for stock in pair] + ['SPY']))
        
        try:
            price_data = data_manager.fetch_stock_data(symbols, start_date, end_date)
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return

        print("\nPreparing training data...")
        training_data = []
        lookback = 60
        
        stock1, stock2 = pairs[0]
        for i in range(lookback, len(price_data)):
            window_data = price_data.iloc[i-lookback:i]
            features = analyzer.create_features(window_data[stock1], window_data[stock2])
            future_return = (price_data[stock2].iloc[i] / price_data[stock2].iloc[i-1] - 
                           price_data[stock1].iloc[i] / price_data[stock1].iloc[i-1])
            success = 1 if future_return > 0 else 0
            training_data.append({
                'features': features,
                'success': success
            })

        X = pd.DataFrame([d['features'] for d in training_data])
        y = pd.Series([d['success'] for d in training_data])

        analyzer.train_model(X, y)

        for stock1, stock2 in pairs:
            print(f"\nAnalyzing pair: {stock1}-{stock2}")
            try:
                success_prob = analyzer.predict_pair_success(
                    price_data[stock1], 
                    price_data[stock2]
                )
                
                print(f"Pair {stock1}-{stock2}:")
                print(f"Predicted success probability: {success_prob:.2f}")

                if success_prob > 0.3:
                    print("Running backtest...")
                    returns = trader.backtest(
                        price_data[stock1], 
                        price_data[stock2]
                    )
                    
                    if len(returns) > 0:
                        print(f"\nBacktest Results:")
                        print(f"Total Return: {returns.sum():.2%}")
                        print(f"Sharpe Ratio: {returns.mean()/returns.std()*np.sqrt(252):.2f}")
                        
                        # Monthly statistics
                        monthly_returns = returns.resample('ME').sum()
                        print(f"Maximum Monthly Return: {monthly_returns.max():.2%}")
                        print(f"Minimum Monthly Return: {monthly_returns.min():.2%}")
                        print(f"% Profitable Months: {(monthly_returns > 0).mean():.2%}")
                        
                        # Drawdown
                        cumulative = (1 + returns).cumprod()
                        running_max = cumulative.expanding().max()
                        drawdown = (cumulative - running_max) / running_max
                        print(f"Maximum Drawdown: {drawdown.min():.2%}")
                else:
                    print("Skipping pair due to low success probability")
                    
            except Exception as e:
                print(f"Error analyzing pair {stock1}-{stock2}: {str(e)}")
                continue

        print("\nAnalysis complete!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()