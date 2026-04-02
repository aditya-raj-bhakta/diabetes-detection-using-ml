import pandas as pd
import numpy as np
import requests
import io
import os
import warnings
from scipy.io import arff

class DataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
    def load_heart_disease(self):
        """
        Loads UCI Heart Disease Dataset (Cleveland)
        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        
        try:
            print(f"Loading Heart Disease data from {url}...")
            response = requests.get(url)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text), header=None, names=columns)
                # target 0 is healthy, 1-4 is disease. Convert to binary.
                df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
                # Handle missing '?' values
                df = df.replace('?', np.nan)
                return df
            else:
                print("Failed to download heart disease data.")
        except Exception as e:
            print(f"Error loading heart disease data: {e}")
            
        return None

    def load_diabetes(self):
        """
        Loads Pima Indians Diabetes Dataset
        """
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        columns = [
            'n_pregnant', 'glucose', 'diastolic_bp', 'skin_thickness', 
            'insulin', 'bmi', 'dpf', 'age', 'target'
        ]
        
        try:
            print(f"Loading Diabetes data from {url}...")
            response = requests.get(url)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text), header=None, names=columns)
                return df
            else:
                print("Failed to download diabetes data.")
        except Exception as e:
            print(f"Error loading diabetes data: {e}")
            
        return None

    def load_kidney_disease(self):
        """
        Loads UCI Chronic Kidney Disease (CKD) Dataset.
        Note: The raw UCI file is ARFF. We'll attempt to load specific CSV version if available, 
        or parse ARFF manually/with scipy.
        """
        # Using a reliable Github mirror of the CSV version
        url = "https://raw.githubusercontent.com/abhinavsagar/Kidney-Disease-Prediction/master/kidney_disease.csv"
        
        try:
            print(f"Loading Kidney Disease data from {url}...")
            response = requests.get(url)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))
                
                # The dataset usually requires significant cleaning
                # Rename 'classification' to 'target'
                if 'classification' in df.columns:
                    df['target'] = df['classification'].map({'ckd': 1, 'notckd': 0, 'ckd\t': 1})
                    df = df.drop(columns=['classification', 'id'], errors='ignore')
                    
                return df
            else:
                print("Failed to download kidney disease data.")
        except Exception as e:
            print(f"Error loading kidney disease data: {e}")
            
        return None

    def load_liver_disease(self):
        """
        Loads Indian Liver Patient Dataset (ILPD)
        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
        columns = [
            'age', 'gender', 'total_bilirubin', 'direct_bilirubin', 'alkaline_phosphotase',
            'alamine_aminotransferase', 'aspartate_aminotransferase', 'total_protiens',
            'albumin', 'albumin_and_globulin_ratio', 'target'
        ]
        
        try:
            print(f"Loading Liver Disease data from {url}...")
            response = requests.get(url)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text), header=None, names=columns)
                
                # Dataset encoding: 1 = liver disease, 2 = no liver disease
                # We want 1 = disease, 0 = no disease
                df['target'] = df['target'].map({1: 1, 2: 0})
                
                # Encode gender
                df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
                
                return df
            else:
                print("Failed to download liver disease data.")
        except Exception as e:
            print(f"Error loading liver disease data: {e}")
            
        return None

if __name__ == "__main__":
    dl = DataLoader()
    
    print("\n--- Testing Data Loader ---")
    heart = dl.load_heart_disease()
    print(f"Heart Data: {heart.shape if heart is not None else 'Failed'}")
    
    diabetes = dl.load_diabetes()
    print(f"Diabetes Data: {diabetes.shape if diabetes is not None else 'Failed'}")
    
    kidney = dl.load_kidney_disease()
    print(f"Kidney Data: {kidney.shape if kidney is not None else 'Failed'}")
    
    liver = dl.load_liver_disease()
    print(f"Liver Data: {liver.shape if liver is not None else 'Failed'}")
