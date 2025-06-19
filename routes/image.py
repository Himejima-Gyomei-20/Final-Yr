# routes/image.py

import os
import pickle
import numpy as np
import base64
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity
import shutil
from deepface import DeepFace

router = APIRouter(tags=["Image Recognition"])

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ────────────────────────────────────────────────────
# Now the dataset folder lives in the same directory as this code:
DATASET_ROOT = "Interpol Red Notices"
# ────────────────────────────────────────────────────

# Load the 128-dimensional embeddings saved earlier
with open("face_embeddings.pkl", "rb") as f:
    embedding_data: dict = pickle.load(f)
    # embedding_data keys are of the form "FOLDER_NAME__FILENAME"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@router.post("/recognize")
async def recognize_person(image: UploadFile = File(...)):
    if not allowed_file(image.filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only JPG, JPEG, PNG allowed."
        )

    # Save uploaded file
    save_path = os.path.join(UPLOAD_FOLDER, image.filename)
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save error: {e}")

    # Extract 128-d embedding for the uploaded image
    try:
        reps = DeepFace.represent(
            img_path=save_path,
            model_name="Facenet",
            enforce_detection=True
        )
        if not reps:
            raise HTTPException(status_code=400, detail="No face detected in the image.")

        test_embedding = np.array(reps[0]["embedding"]).reshape(1, -1)  # (1,128)
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"DeepFace error: {ex}")

    # Find the best‐match among stored embeddings
    best_match_key = None
    best_score = -1.0
    threshold = 0.65

    for image_key, saved_emb in embedding_data.items():
        score = cosine_similarity(test_embedding, saved_emb)[0][0]
        if score > best_score:
            best_score = float(score)
            best_match_key = image_key

    if best_score < threshold:
        return JSONResponse({
            "person": "not in our Criminal Database",
            "confidence": best_score
        })

    # Split “FOLDER_NAME__FILENAME” into folder_name + filename
    try:
        folder_name, filename = best_match_key.split("__", 1)
    except ValueError:
        folder_name = best_match_key
        filename = ""

    # Build path to the matched image inside ./Interpol Red Notices/
    matched_image_path = os.path.join(DATASET_ROOT, folder_name, filename)

    if not os.path.isfile(matched_image_path):
        return JSONResponse({
            "person": folder_name,
            "confidence": best_score,
            "image_base64": None,
            "warning": f"Matched file not found on disk: {matched_image_path}"
        })

    # Read and Base64‐encode the matched image
    try:
        with open(matched_image_path, "rb") as img_f:
            img_bytes = img_f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading matched image: {e}")

    return JSONResponse({
        "person": folder_name,
        "confidence": best_score,
        "image_base64": img_b64
    })
