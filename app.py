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

from ohm import (
    load_dnn_model,
    load_normalization_params,
    load_regressor,
    Wav2Vec2FeatureExtractorModel,
    predict_ohm_rating,
)

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


class DNNModel(nn.Module):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 1024)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x


def test_predict_ohm_rating(temp_folder):
    """Tests the predict_ohm_rating function from ohm.py."""
    # Define paths
    speaker_folder_path = temp_folder
    model_path = './models/english_xlsr_librispeech_model_32batch_100h_valid.pth'
    regressor_path = './models/regressor_model.pkl'
    mean_path = './models/Mean.npy'
    std_path = './models/Std.npy'

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models and parameters
    print("Loading DNN model...")
    dnn_model = load_dnn_model(model_path, device)

    print("Loading normalization parameters...")
    feats_m1, feats_s1 = load_normalization_params(mean_path, std_path)

    print("Loading regressor model...")
    regressor = load_regressor(regressor_path)

    print("Loading Wav2Vec2 feature extractor model...")
    wav2vec_model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53")
    feature_extractor_model = Wav2Vec2FeatureExtractorModel(
        wav2vec_model).to(device)
    feature_extractor_model.eval()

    # Run prediction
    print("Predicting perceptual OHM rating...")
    perceptual_rating = predict_ohm_rating(
        speaker_folder_path, feature_extractor_model, dnn_model, feats_m1, feats_s1, regressor, device
    )

    # Check the output
    if perceptual_rating is not None:
        print(
            f"Test Passed! Predicted Perceptual OHM Rating: {perceptual_rating:.2f}")
    else:
        print("Test Failed: No valid OHM scores computed for the speaker.")

    return perceptual_rating

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
    upload_file_name = "66b484e6-1579-40ad-a052-99a4d34bfd8a-2024-10-22T21:26:14.714Z-25"
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
