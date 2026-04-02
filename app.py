from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import os
import io
import pandas as pd
from framework import MultiDiseasePredictor

app = FastAPI(title="Multi-Disease Predictor")

# Setup templates
if not os.path.exists("templates"):
    os.makedirs("templates")
templates = Jinja2Templates(directory="templates")

# Initialize Predictor Global Variable
# In a production app, we would load pickles. 
# Here, to ensure it works with the current code structure, we'll initialize it.
# If models exist in 'models/' directory, we should modify framework to load them, 
# otherwise we train on startup.
print("Initializing Framework...")
predictor = MultiDiseasePredictor()

# Check if models are trained by looking for files, otherwise train
# (For prototype simplicity, we will trigger training if models dir is empty or just let it train)
# The framework.py as written trains on demand. We will run training once on startup.
print("Training/Loading Models... This may take a moment.")
predictor.train_all()
print("Models Ready!")

class PatientData(BaseModel):
    age: float
    sex: int
    cp: float = 0
    trestbps: float = 0
    chol: float = 0
    fbs: float = 0
    restecg: float = 0
    thalach: float = 0
    exang: float = 0
    oldpeak: float = 0
    slope: float = 0
    ca: float = 0
    thal: float = 0
    glucose: float = 0
    diastolic_bp: float = 0
    skin_thickness: float = 0
    insulin: float = 0
    bmi: float = 0
    dpf: float = 0
    n_pregnant: float = 0
    total_bilirubin: float = 0
    direct_bilirubin: float = 0
    alkaline_phosphotase: float = 0
    alamine_aminotransferase: float = 0
    aspartate_aminotransferase: float = 0
    total_protiens: float = 0
    albumin: float = 0
    albumin_and_globulin_ratio: float = 0
    # Add other fields as needed for all datasets

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(data: PatientData):
    try:
        # Convert Pydantic model to dict
        patient_dict = data.dict()
        
        # Run prediction
        results = predictor.predict_patient(patient_dict)
        
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
