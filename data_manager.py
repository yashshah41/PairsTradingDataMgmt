import sqlite3
import yfinance as yf
import pandas as pd
import os

class DataManager:
    def __init__(self, db_path='data/stock_data.db'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.initialize_db()

    def initialize_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS stock_prices
                    (date TEXT, symbol TEXT, price REAL, 
                    PRIMARY KEY (date, symbol))''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS trading_pairs
                    (pair_id INTEGER PRIMARY KEY,
                    stock1 TEXT, stock2 TEXT,
                    correlation REAL)''')
        
        conn.commit()
        conn.close()

    def fetch_stock_data(self, symbols, start_date, end_date):
        data = {}
        print(f"Fetching data for {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols, 1):
            print(f"Fetching {symbol} ({i}/{len(symbols)})...")
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                if hist.empty:
                    print(f"Warning: No data received for {symbol}")
                    continue
                    
                data[symbol] = hist['Close']
                
                conn = sqlite3.connect(self.db_path)
                df = pd.DataFrame(hist['Close'])
                df.reset_index(inplace=True)
                
                for _, row in df.iterrows():
                    conn.execute('INSERT OR REPLACE INTO stock_prices (date, symbol, price) VALUES (?, ?, ?)',
                               (row['Date'].strftime('%Y-%m-%d'), symbol, row['Close']))
                conn.commit()
                conn.close()
                print(f"Successfully processed {symbol}")
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                continue
                
        if not data:
            raise Exception("No data was successfully fetched for any symbol")
            
        return pd.DataFrame(data)