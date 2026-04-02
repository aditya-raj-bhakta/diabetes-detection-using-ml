import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import pickle
import os

class DiseaseClassifier:
    def __init__(self, disease_name, output_dir='models'):
        self.disease_name = disease_name
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.best_model = None
        self.best_score = 0
        
    def get_param_grids(self):
        """
        Returns a dictionary of models and their hyperparameter grids.
        """
        grids = {
            'LogisticRegression': {
                'model': LogisticRegression(solver='liblinear', random_state=42),
                'params': {
                    'C': [0.01, 0.1, 1, 10], 
                    'penalty': ['l1', 'l2']
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear', 'poly']
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
            }
        }
        return grids

    def train(self, X, y):
        """
        Iterates through 4 model types, performs GridSearchCV, and selects the best one for this disease.
        """
        print(f"\nTraining models for {self.disease_name}...")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grids = self.get_param_grids()
        
        best_overall_model = None
        best_overall_score = -1
        best_model_name = ""
        
        for name, config in grids.items():
            print(f"  Tuning {name}...")
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['params'],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            try:
                grid_search.fit(X, y)
                score = grid_search.best_score_
                print(f"    Best {name} Score (ROC-AUC): {score:.4f}")
                
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_model = grid_search.best_estimator_
                    best_model_name = name
            except Exception as e:
                print(f"    Failed to train {name}: {e}")
                
        self.best_model = best_overall_model
        print(f"Winner for {self.disease_name}: {best_model_name} with ROC-AUC: {best_overall_score:.4f}")
        
        # Save model
        save_path = os.path.join(self.output_dir, f"{self.disease_name}_model.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(self.best_model, f)
            
        return self.best_model

    def evaluate(self, X_test, y_test):
        """
        Evaluates the best model on test set.
        """
        if self.best_model is None:
            print("No model trained yet.")
            return {}
            
        y_pred = self.best_model.predict(X_test)
        y_prob = self.best_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
        
        return metrics

    def predict_proba(self, X):
        if self.best_model:
            return self.best_model.predict_proba(X)[:, 1]
        return np.zeros(X.shape[0])
