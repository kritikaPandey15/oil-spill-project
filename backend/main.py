from fastapi import FastAPI, UploadFile, File, Form
from models import InvestigationResult, InvestigationRequest

from services.ml_service import detect_spill
from services.drift_service import estimate_origin
from services.ais_service import find_vessels
import os
import uuid

app = FastAPI(
    title="Oil Spill Investigation API",
    description="Backend for SIH26143",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Oil Spill Investigation API is running"
    }


@app.post("/api/investigate", response_model=InvestigationResult)
async def investigate(
    latitude: float = Form(...),
    longitude: float = Form(...),
    timestamp: str = Form(...),
    image: UploadFile = File(...)
):
    # Create uploads folder if it doesn't exist
    os.makedirs("uploads", exist_ok=True)

    # Create a unique filename
    filename = f"{uuid.uuid4()}_{image.filename}"
    image_path = os.path.join("uploads", filename)

    # Save uploaded image
    image_data = await image.read()

    with open(image_path, "wb") as file:
        file.write(image_data)

    # Step 1: Detect oil spill
    spill = detect_spill(image_path)

    # Step 2: Estimate origin
    origin = estimate_origin(
        latitude,
        longitude
    )

    # Step 3: Find nearby vessels
    vessels = find_vessels(
        origin["latitude"],
        origin["longitude"]
    )

    return {
        "spill": spill,
        "origin": origin,
        "vessels": vessels
    }