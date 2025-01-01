from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr
import boto3
import os
from tempfile import NamedTemporaryFile
from pathlib import Path
from dotenv import load_dotenv
from transformers import Wav2Vec2Model
import torch
import uvicorn
import torch.nn as nn
from mailjet_rest import Client
from typing import List

from ohm import predict_ohm_rating

load_dotenv()
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


class EmailSchema(BaseModel):
    subject: str
    body: str


async def send_email(email: EmailSchema):
    api_key = os.environ.get('MAILJET_API_KEY')
    api_secret = os.environ.get('MAILJET_API_SECRET')
    mailjet = Client(auth=(api_key, api_secret), version='v3.1')

    data = {
        'Messages': [
            {
                "From": {
                    "Email": os.environ.get("EMAIL_FROM"),
                    "Name": os.environ.get("EMAIL_FROM_NAME")
                },
                "To": [
                    {
                        "Email": "kkothadi@asu.edu",
                        "Name": "Cleft Care User"
                    },
                    {
                        "Email": "alambat@asu.edu",
                        "Name": "Cleft Care Admin"
                    }
                ],
                "Subject": email.subject,
                "HTMLPart": email.body
            }
        ]
    }

    try:
        result = mailjet.send.create(data=data)
        if result.status_code == 200:
            print(f"Email sent successfully: {result.json()}")
        else:
            print(
                f"Failed to send email: {result.status_code}, {result.json()}")
    except Exception as e:
        print(f"Exception occurred while sending email: {e}")

"""
Health Check Endpoint
"""


@app.get("/")
async def health_check():
    return {"status": "ok"}


@app.get("/predict-test")
async def predict_ohm(background_tasks: BackgroundTasks):
    upload_file_name = "MILE_03_SP_0002_UTT_0003.wav"
    # Generate S3 file path based on userId
    s3_key = upload_file_name
    name = "Test User"
    user_id = "undefined"
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
        perceptual_rating = predict_ohm_rating(temp_folder)

        # Prepare email data
        email = EmailSchema(
            subject="CleftCare: New Prediction Result",
            body=(
                f"<p>The patient {user_id} and {name} was recorded in ASU by the community worker Krupa.</p>"
                f"<p>The level of hypernasality in its speech as per cleft care is {perceptual_rating}.</p>"
            )
        )

        # Add email sending to background tasks
        background_tasks.add_task(send_email, email)

        return {"perceptualRating": perceptual_rating}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing file: {str(e)}")


"""
Predict Endpoint:
- Accepts a POST request with userId and uploadFileName
"""


class PredictRequest(BaseModel):
    userId: str
    name: str
    promptNumber: int
    uploadFileName: str


@app.post("/predict")
async def predict_ohm(request: PredictRequest, background_tasks: BackgroundTasks):
    user_id = request.userId
    name = request.name
    prompt_number = request.promptNumber
    upload_file_name = request.uploadFileName
    print(
        f"Received request for userId: {user_id}, uploadFileName: {upload_file_name}")
    # Generate S3 file path based on userId
    s3_key = upload_file_name  # Customize if needed

    try:
        audio_dir = os.path.join(os.getcwd(), "audios")
        os.makedirs(audio_dir, exist_ok=True)
        # Download file from S3
        with NamedTemporaryFile(delete=True, suffix=".m4a", dir=audio_dir) as temp_file:
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
        perceptual_rating = predict_ohm_rating(temp_folder)
        # Prepare email data
        email = EmailSchema(
            subject="CleftCare: New Prediction Result",
            body=(
                f"<p>The patient {user_id} and {name} was recorded in ASU by the community worker Krupa.</p>"
                f"<p>The level of hypernasality in its speech as per cleft care is {perceptual_rating}.</p>"
            )
        )

        # Add email sending to background tasks
        background_tasks.add_task(send_email, email)
        # Delete the .wav file if it was created
        if os.path.exists(wav_path) and wav_path != temp_file_path:
            os.remove(wav_path)
            print(f"Deleted .wav file: {wav_path}")
        return {"perceptualRating": perceptual_rating}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing file: {str(e)}")


if __name__ == "__main__":
    # Read the PORT environment variable, default to 8080 if not set
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
