from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import os
from tempfile import NamedTemporaryFile
from pathlib import Path
from dotenv import load_dotenv
from transformers import Wav2Vec2Model
import torch
import torch.nn as nn

from ohm import predict_ohm_rating

# Initialize FastAPI app
app = FastAPI()

# S3 Client Setup
S3_BUCKET_NAME = 'cleftcare-test'
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)


# Health Check
@app.get("/")
async def health_check():
    return {"status": "ok"}

# Request Model


class PredictRequest(BaseModel):
    userId: str
    uploadFileName: str

# GET Endpoint


@app.get("/predict-test")
async def predict_ohm():
    upload_file_name = "MILE_03_SP_0002_UTT_0003.wav"
    # Generate S3 file path based on userId
    # s3_key = f"{upload_file_name}.m4a"  # Customize if needed
    s3_key = upload_file_name

    try:
        audio_dir = os.path.join(os.getcwd(), "audios")
        os.makedirs(audio_dir, exist_ok=True)
        # Download file from S3
        with NamedTemporaryFile(delete=False, suffix=".m4a", dir=audio_dir) as temp_file:
            temp_file_path = temp_file.name
            s3.download_file(S3_BUCKET_NAME, s3_key, temp_file_path)
            print("Downloaded file from S3: ", temp_file_path)

            # Check if the file is already in .wav format
            if not temp_file_path.endswith(".wav"):
                # Convert .m4a to .wav
                wav_path = temp_file_path.replace(".m4a", ".wav")
                os.system(f"ffmpeg -i {temp_file_path} {wav_path}")
            else:
                wav_path = temp_file_path

        # Process the downloaded file for prediction
        temp_folder = Path(temp_file_path).parent
        perceptual_rating = predict_ohm_rating_old(temp_folder)

        return {"perceptualRating": perceptual_rating}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing file: {str(e)}")


# POST Endpoint


@app.post("/predict")
async def predict_ohm(request: PredictRequest):
    user_id = request.userId
    upload_file_name = request.uploadFileName

    # Generate S3 file path based on userId
    s3_key = f"{upload_file_name}.m4a"  # Customize if needed

    try:
        # Download file from S3
        with NamedTemporaryFile(delete=False, suffix=".m4a") as temp_file:
            temp_file_path = temp_file.name
            s3.download_file(S3_BUCKET_NAME, s3_key, temp_file_path)

        # Convert .m4a to .wav
        wav_path = temp_file_path.replace(".m4a", ".wav")
        os.system(f"ffmpeg -i {temp_file_path} {wav_path}")

        # Process the downloaded file for prediction
        temp_folder = Path(temp_file_path).parent
        perceptual_rating = test_predict_ohm_rating(temp_folder)

        return {"userId": user_id, "perceptualRating": perceptual_rating}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing file: {str(e)}")
