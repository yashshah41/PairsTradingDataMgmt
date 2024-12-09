from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

class PairsAnalyzer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def create_features(self, prices1, prices2):
        correlation = prices1.corr(prices2)
        
        returns1 = prices1.pct_change().dropna()
        returns2 = prices2.pct_change().dropna()
        
        volatility1 = returns1.std()
        volatility2 = returns2.std()
        
        spread = prices2 - (prices2.mean()/prices1.mean()) * prices1
        spread_vol = spread.std()
        
        return pd.Series({
            'correlation': correlation,
            'vol_ratio': volatility1/volatility2,
            'spread_vol': spread_vol
        })

    def train_model(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True
        print("Model training completed")

    def predict_pair_success(self, prices1, prices2):
        if not self.is_trained:
            raise Exception("Model needs to be trained before making predictions")
            
        features = self.create_features(prices1, prices2)
        features_df = pd.DataFrame([features])
        return self.model.predict_proba(features_df)[0][1]