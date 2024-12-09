import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

class PairsTrader:
    def __init__(self, window=60, entry_threshold=1.5, exit_threshold=0.5, stop_loss=-0.03):
        self.window = window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.max_position_hold = 20

    def analyze_pair(self, prices1, prices2):
    # Convert prices to returns for correlation calculation
        df = pd.DataFrame({
            'p1': prices1,
            'p2': prices2
        }).dropna()
        
        # Calculate returns
        returns1 = df['p1'].pct_change().dropna()
        returns2 = df['p2'].pct_change().dropna()
        
        # Calculate correlation using returns instead of prices
        correlation = returns1.corr(returns2)
        
        print(f"\nPair Analysis:")
        print(f"Correlation: {correlation:.3f}")
        print(f"Number of observations: {len(df)}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        print(f"\nPrice Statistics:")
        print(f"Stock 1 mean: {df['p1'].mean():.2f}")
        print(f"Stock 2 mean: {df['p2'].mean():.2f}")
        print(f"Stock 1 std: {df['p1'].std():.2f}")
        print(f"Stock 2 std: {df['p2'].std():.2f}")
        
        # Calculate rolling correlation to show stability
        rolling_corr = returns1.rolling(window=60).corr(returns2)
        print(f"Average 60-day rolling correlation: {rolling_corr.mean():.3f}")

        return correlation > 0.5

    def calculate_hedge_ratio(self, prices1, prices2, window=60):
        df = pd.DataFrame({'p1': prices1, 'p2': prices2}).dropna()
        hedge_ratios = []
        
        for i in range(window, len(df)):
            window_data = df.iloc[i-window:i]
            model = OLS(window_data['p2'], sm.add_constant(window_data['p1']))
            results = model.fit()
            hedge_ratios.append(results.params.iloc[1])
            
        return pd.Series(hedge_ratios, index=df.index[window:])

    def calculate_spread(self, prices1, prices2, hedge_ratio):
        df = pd.DataFrame({
            'p1': prices1,
            'p2': prices2,
            'ratio': hedge_ratio
        }).dropna()
        return df['p2'] - df['ratio'] * df['p1']

    def generate_signals(self, prices1, prices2):
        if not self.analyze_pair(prices1, prices2):
            print("Pair does not meet correlation criteria")
            return None, None
            
        hedge_ratio = self.calculate_hedge_ratio(prices1, prices2)
        print(f"Average Hedge Ratio: {hedge_ratio.mean():.3f}")
        
        spread = self.calculate_spread(prices1, prices2, hedge_ratio)
        
        zscore = pd.Series(index=spread.index)
        for i in range(self.window, len(spread)):
            window_spread = spread.iloc[i-self.window:i]
            zscore.iloc[i] = (spread.iloc[i] - window_spread.mean()) / window_spread.std()
        
        signals = pd.DataFrame(index=spread.index)
        signals['zscore'] = zscore
        signals['long_entry'] = zscore < -self.entry_threshold
        signals['short_entry'] = zscore > self.entry_threshold
        signals['exit'] = abs(zscore) < self.exit_threshold
        
        aligned_prices = pd.DataFrame({
            'p1': prices1,
            'p2': prices2
        }).loc[signals.index]
        
        return signals, aligned_prices

    def backtest(self, prices1, prices2):
        signals, aligned_prices = self.generate_signals(prices1, prices2)
        
        if signals is None:
            return pd.Series()
        
        position = 0
        returns = []
        position_count = 0
        cumulative_return = 0
        trade_stats = []
        
        for i in range(1, len(signals)):
            ret1 = (aligned_prices['p1'].iloc[i] - aligned_prices['p1'].iloc[i-1]) / aligned_prices['p1'].iloc[i-1]
            ret2 = (aligned_prices['p2'].iloc[i] - aligned_prices['p2'].iloc[i-1]) / aligned_prices['p2'].iloc[i-1]
            
            if position != 0:
                trade_return = position * (ret2 - ret1)
                cumulative_return += trade_return
                position_count += 1
                
                if (cumulative_return < self.stop_loss or 
                    position_count > self.max_position_hold or 
                    signals['exit'].iloc[i]):
                    
                    trade_stats.append({
                        'return': cumulative_return,
                        'duration': position_count,
                        'exit_type': 'stop_loss' if cumulative_return < self.stop_loss else 
                                   'time_limit' if position_count > self.max_position_hold else 
                                   'signal'
                    })
                    position = 0
                    cumulative_return = 0
                    position_count = 0
            
            if position == 0:
                if signals['long_entry'].iloc[i]:
                    position = 1
                elif signals['short_entry'].iloc[i]:
                    position = -1
            
            returns.append(position * (ret2 - ret1))
        
        returns_series = pd.Series(returns, index=signals.index[1:])
        
        self.print_trade_statistics(trade_stats, returns_series)
        return returns_series

    def print_trade_statistics(self, trade_stats, returns):
        print("\nDetailed Performance Statistics:")
        print(f"Total trades: {len(trade_stats)}")
        
        if trade_stats:
            returns_list = [t['return'] for t in trade_stats]
            print(f"Win rate: {sum(r > 0 for r in returns_list)/len(returns_list):.2%}")
            print(f"Average trade return: {np.mean(returns_list):.2%}")
            
            winning_trades = [r for r in returns_list if r > 0]
            losing_trades = [r for r in returns_list if r < 0]
            
            if winning_trades:
                print(f"Average winning trade: {np.mean(winning_trades):.2%}")
            if losing_trades:
                print(f"Average losing trade: {np.mean(losing_trades):.2%}")