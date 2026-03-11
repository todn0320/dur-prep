import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from src.pipeline.run_pipeline import run_pipeline

app = FastAPI(title="Pill Identification API")


@app.get("/")
def root():
    return {"message": "Pill Identification API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="파일이 없습니다.")

    os.makedirs("temp", exist_ok=True)
    temp_path = os.path.join("temp", file.filename)

    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        result = run_pipeline(temp_path)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)