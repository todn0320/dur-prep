from fastapi import FastAPI
from src.pipeline.run_pipeline import run_pipeline

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Pill AI API is running"}

@app.get("/predict")
def predict(image_path: str):
    return run_pipeline(image_path)