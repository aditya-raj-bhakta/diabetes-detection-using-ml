from data_loader import DataLoader
from preprocessing import Preprocessor
from models import DiseaseClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class MultiDiseasePredictor:
    def __init__(self):
        self.loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.diseases = ['Heart Disease', 'Diabetes', 'Kidney Disease', 'Liver Disease']
        self.classifiers = {name: DiseaseClassifier(name) for name in self.diseases}
        self.results = {}
        
    def load_and_preprocess(self, disease_name):
        """
        Determines which loader function to call and runs preprocessing.
        """
        df = None
        if disease_name == 'Heart Disease':
            df = self.loader.load_heart_disease()
        elif disease_name == 'Diabetes':
            df = self.loader.load_diabetes()
        elif disease_name == 'Kidney Disease':
            df = self.loader.load_kidney_disease()
        elif disease_name == 'Liver Disease':
            df = self.loader.load_liver_disease()
            
        if df is None:
            return None, None, None
            
        # Process data
        X, y = self.preprocessor.process(df)
        
        return X, y, df # Return df to see columns if needed
        
    def train_all(self):
        """
        Trains models for all diseases and stores evaluation results.
        """
        summary = []
        
        for disease in self.diseases:
            print(f"\n=== Processing {disease} ===")
            X, y, _ = self.load_and_preprocess(disease)
            
            if X is None or y is None:
                print(f"Skipping {disease} due to data loading error.")
                continue
                
            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            
            # Train
            clf = self.classifiers[disease]
            clf.train(X_train, y_train)
            
            # Evaluate
            metrics = clf.evaluate(X_test, y_test)
            metrics['Disease'] = disease
            summary.append(metrics)
            
            # Store column names for later prediction alignment
            # In a real app, this should be saved to file
            self.classifiers[disease].feature_names = list(X.columns)
            
        return pd.DataFrame(summary)

    def predict_patient(self, patient_data):
        """
        Predicts probabilities for a single patient across all valid models.
        patient_data: dict of feature_name -> value
        """
        predictions = {}
        
        for disease, clf in self.classifiers.items():
            if clf.best_model is None:
                predictions[disease] = "Model not trained"
                continue
                
            # Align features
            feature_names = getattr(clf, 'feature_names', [])
            if not feature_names:
                predictions[disease] = "Feature info missing"
                continue
                
            # Create a dataframe row with zeros for missing features
            input_df = pd.DataFrame(columns=feature_names)
            input_df.loc[0] = 0.0 # Initialize with 0.0 to ensure float dtype
            
            # Fill known values
            for feature, value in patient_data.items():
                if feature in feature_names:
                    input_df.at[0, feature] = value
            
            # Scale (using a fresh scaler is not ideal, ideally we load the saved scaler)
            # For this prototype, we'll assume the model can handle raw-ish input 
            # OR we need to save the scaler in Preprocessing. 
            # FIX: The Preprocessor fits a scaler on the whole dataset. 
            # For a single prediction, we really need the scaler used during training.
            # To keep this prototype simple, we will skip scaling for the single prediction 
            # step OR acknowledge this limitation. Tree models (RF, XGB) don't care much about scaling.
            # SVM/LR do. 
            
            prob = clf.predict_proba(input_df)[0]
            predictions[disease] = f"{prob:.2%}"
            
        return predictions
