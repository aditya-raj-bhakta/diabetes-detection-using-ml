import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

class Preprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        
    def clean_data(self, df):
        """
        Basic cleaning: converting columns to numeric, dropping duplicates.
        """
        if df is None:
            return None
            
        print(f"Original shape: {df.shape}")
        
        # Drop duplicates
        df = df.drop_duplicates()
        
        # Convert all columns to numeric, forcing errors to NaN
        # (Target column should already be numeric from loader, but good to be safe)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df

    def handle_missing_values(self, X, y=None):
        """
        Imputes missing values using median strategy.
        """
        if X is None or X.shape[0] == 0:
            return X, y

        # Fit on X and transform
        X_imputed = self.imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=X.columns)
        
        # If y has missing values (rare for target), drop those rows
        if y is not None:
            # Align indices since X was recreated
            y = y.reset_index(drop=True)
            mask = ~np.isnan(y)
            X = X[mask]
            y = y[mask]
            
        return X, y

    def scale_features(self, X):
        """
        Scales features using StandardScaler.
        """
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)

    def balance_classes(self, X, y):
        """
        Uses SMOTE to balance classes if there is a significant imbalance.
        """
        counter = Counter(y)
        print(f"Original class distribution: {counter}")
        
        # Heuristic: only apply SMOTE if minority class is < 40% of majority
        counts = list(counter.values())
        if len(counts) < 2:
            return X, y # Can't balance 1 class
            
        minority_ratio = min(counts) / max(counts)
        
        if minority_ratio < 0.6: # If imbalance is bad enough
            print("Applying SMOTE...")
            try:
                smote = SMOTE(random_state=42)
                X_res, y_res = smote.fit_resample(X, y)
                print(f"Resampled class distribution: {Counter(y_res)}")
                return X_res, y_res
            except Exception as e:
                print(f"SMOTE failed (possibly too few samples): {e}")
                return X, y
        
        return X, y

    def process(self, df, target_col='target'):
        """
        Full pipeline: Clean -> Split -> Impute -> Scale -> Balance
        Returns: X_final, y_final
        """
        df = self.clean_data(df)
        
        if target_col not in df.columns:
            print(f"Target column '{target_col}' not found.")
            return None, None
            
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Impute
        X, y = self.handle_missing_values(X, y)
        
        # Scale
        X = self.scale_features(X)
        
        # Balance
        X, y = self.balance_classes(X, y)
        
        return X, y
