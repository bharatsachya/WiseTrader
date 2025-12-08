import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class StockLogisticModel:
    def __init__(self, df):
        self.df = df.copy()
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.X_test = None
        self.y_test = None
        self.y_pred = None

    def prepare_data(self):
        """
        1. Creates a binary target: 1 if Tomorrow's Close > Today's Close, else 0.
        2. Aligns features and target.
        """
        # Create Target: Did the price go up the NEXT day?
        # shift(-1) looks at the next row.
        self.df['Target'] = (self.df['close'].shift(-1) > self.df['close']).astype(int)

        # Drop the last row because it has no 'Target' (we don't know tomorrow's price yet)
        self.df.dropna(inplace=True)

        # Define Feature Columns (Exclude raw price and target)
        # Assuming you used the previous script, we use the technical indicators
        feature_cols = [c for c in self.df.columns if c not in ['target', 'date', 'Target']]
        
        X = self.df[feature_cols]
        y = self.df['Target']
        
        return X, y

    def train(self, test_size=0.2):
        """
        Splits data SEQUENTIALLY (no shuffling) and trains the model.
        """
        X, y = self.prepare_data()

        # IMPORTANT: shuffle=False prevents "looking into the future" (Data Leakage)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        # Scale features (Logistic Regression is sensitive to magnitude)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train
        self.model.fit(X_train_scaled, y_train)

        # Save for evaluation
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        print("Model Trained Successfully.")

    def evaluate(self):
        """
        Prints accuracy metrics and generates a confusion matrix.
        """
        if self.X_test is None:
            print("Run train() first.")
            return

        self.y_pred = self.model.predict(self.X_test)
        
        print("\n--- Model Evaluation ---")
        print(f"Accuracy: {accuracy_score(self.y_test, self.y_pred):.2f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred))

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Simple text visualization of Confusion Matrix
        print("\nConfusion Matrix:")
        print(f"True Negatives (Correctly Predicted DOWN): {cm[0][0]}")
        print(f"False Positives (Predicted UP, actually DOWN): {cm[0][1]}")
        print(f"False Negatives (Predicted DOWN, actually UP): {cm[1][0]}")
        print(f"True Positives (Correctly Predicted UP): {cm[1][1]}")

    def predict_proba(self):
        """
        Returns the probability of the price going UP.
        Useful for setting confidence thresholds (e.g., only trade if prob > 60%).
        """
        # Get probability of class 1 (UP)
        probs = self.model.predict_proba(self.X_test)[:, 1]
        return probs

# ==========================================
# Example Usage (Integration with previous step)
# ==========================================
if __name__ == "__main__":
    # 1. Reuse the 'df_features' from the previous step (or create dummy data)
    # This block generates dummy data if you run this script alone
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="D")
    df_features = pd.DataFrame({
        'close': np.random.uniform(100, 200, 1000),
        'rsi': np.random.uniform(30, 70, 1000),
        'macd': np.random.uniform(-1, 1, 1000),
        'sma_20': np.random.uniform(100, 200, 1000),
        'log_return': np.random.normal(0, 0.01, 1000)
    })

    # 2. Initialize and Train
    log_model = StockLogisticModel(df_features)
    log_model.train(test_size=0.2)
    
    # 3. Evaluate
    log_model.evaluate()

    # 4. Check Probabilities (Advanced Usage)
    probabilities = log_model.predict_proba()
    print(f"\nSample Prediction Probabilities (First 5 days of Test Data): {probabilities[:5]}")
