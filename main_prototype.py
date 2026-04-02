from framework import MultiDiseasePredictor
import pandas as pd
import warnings

# Suppress warnings for cleaner output in prototype
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("   MULTI-DISEASE PREDICTION FRAMEWORK PROTOTYPE")
    print("="*60)
    print("DISCLAIMER: Not a diagnostic tool. For academic use only.")
    print("="*60)
    
    predictor = MultiDiseasePredictor()
    
    # 1. Train Models
    print("\n[Phase 1] Loading Data, Preprocessing, and Training Models...")
    results_df = predictor.train_all()
    
    # 2. Show Evaluation Results
    print("\n[Phase 2] Evaluation Results:")
    if not results_df.empty:
        print(results_df.round(4).to_string(index=False))
        
        # Save results
        results_df.to_csv("evaluation_results.csv", index=False)
        print("\nResults saved to 'evaluation_results.csv'.")
    else:
        print("No models were successfully trained. Check internet connection or data sources.")
        return

    # 3. Mock Prediction (Demonstration)
    print("\n[Phase 3] Unified Prediction Demonstration (Mock Patient)")
    
    # Mock patient with values near "unhealthy" ranges just to see probabilities
    mock_patient = {
        'age': 65, 
        'sex': 1, 
        'glucose': 180, # High
        'bmi': 32,      # High
        'chol': 250,    # High
        'trestbps': 140, # High BP
        'total_bilirubin': 1.5,
        'albumin': 2.5
    }
    
    print(f"Patient Data: {mock_patient}")
    probs = predictor.predict_patient(mock_patient)
    
    print("\nDisease Probabilities:")
    for disease, prob in probs.items():
        print(f"  - {disease}: {prob}")
        
    print("\n" + "="*60)
    print("Prototyping Complete.")
    print("="*60)

if __name__ == "__main__":
    main()
