from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import boto3
import os
import subprocess
import re
from tempfile import NamedTemporaryFile
from pathlib import Path
from dotenv import load_dotenv
from transformers import Wav2Vec2Model
import torch
import uvicorn
import torch.nn as nn
from mailjet_rest import Client
from typing import List
import uuid
import time

from ohm import predict_ohm_rating
from gop_module import compute_gop

load_dotenv()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI()

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure with specific domains in production
    allow_credentials=False,  # Set to True if you need credentials
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Input validation functions
def validate_upload_filename(filename: str) -> str:
    """Validate and sanitize upload filename to prevent path traversal"""
    if not filename:
        raise ValueError("Filename cannot be empty")

    # Remove any path separators and relative path indicators
    filename = os.path.basename(filename)
    filename = filename.replace('..', '')

    # Allow only alphanumeric, hyphens, underscores, and dots
    if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
        raise ValueError("Invalid filename format")

    # Check file extension (only allow expected audio formats)
    allowed_extensions = {'.m4a', '.wav', '.mp3', '.flac'}
    _, ext = os.path.splitext(filename.lower())
    if ext not in allowed_extensions:
        raise ValueError(f"Invalid file extension. Allowed: {', '.join(allowed_extensions)}")

    return filename

# Authentication middleware
def verify_api_key(x_api_key: str = Header(None)) -> str:
    """Verify API key from request headers"""
    expected_api_key = os.getenv('API_KEY')

    if not expected_api_key:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error"
        )

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required"
        )

    if x_api_key != expected_api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return x_api_key

# Request ID generation
def generate_request_id() -> str:
    """Generate a unique request ID for tracking"""
    return str(uuid.uuid4())

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
                    # {
                    #     "Email": "kkothadi@asu.edu",
                    #     "Name": "Cleft Care User"
                    # },
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
    # Generate S3 file path based on userId
    user_id = "undefined"
    name = "Test User"
    upload_file_name = "a39ff7d5-b200-4590-83db-ca562f217ec7-1738176080542-25-1.m4a"
    language = "en"
    # user_id = request.userId
    # name = request.name
    # prompt_number = request.promptNumber
    # upload_file_name = request.uploadFileName
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
                result = subprocess.run([
                    "ffmpeg", "-i", temp_file_path, wav_path
                ], capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
            else:
                wav_path = temp_file_path

        # Process the downloaded file for prediction
        temp_folder = Path(temp_file_path).parent
        perceptual_rating = predict_ohm_rating(temp_folder, language)
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


"""
Predict Endpoint:
- Accepts a POST request with userId and uploadFileName
"""


class PredictRequest(BaseModel):
    userId: str
    name: str
    communityWorkerName: str
    promptNumber: int
    language: str
    uploadFileName: str
    sendEmail: bool


class GOPRequest(BaseModel):
    userId: str
    name: str
    communityWorkerName: str
    transcript: str
    uploadFileName: str
    sendEmail: bool


@app.post("/ohm")
@limiter.limit("50/minute")
async def predict_ohm(request: Request, background_tasks: BackgroundTasks, predict_request: PredictRequest, api_key: str = Depends(verify_api_key)):
    # Generate request ID for tracking
    request_id = generate_request_id()
    start_time = time.time()

    user_id = predict_request.userId
    name = predict_request.name
    community_worker_name = predict_request.communityWorkerName
    prompt_number = predict_request.promptNumber
    language = predict_request.language
    upload_file_name = predict_request.uploadFileName
    send_email_flag = predict_request.sendEmail

    # Validate and sanitize filename
    try:
        upload_file_name = validate_upload_filename(upload_file_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {str(e)}")

    print(f"[{request_id}] Received OHM request for userId: {user_id}")
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
                result = subprocess.run([
                    "ffmpeg", "-i", temp_file_path, wav_path
                ], capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
            else:
                wav_path = temp_file_path

        # Process the downloaded file for prediction
        temp_folder = Path(temp_file_path).parent
        perceptual_rating = predict_ohm_rating(temp_folder, language)

        # Only send email if sendEmail is True
        if send_email_flag:
            # Prepare email data
            email = EmailSchema(
                subject="CleftCare: New Prediction Result",
                body=(
                    f"<p>The patient {name} recorded by the community worker {community_worker_name}.</p>"
                    f"<p>The link to the more details for the patient - https://cleftcare-dashboard.vercel.app/dashboard/{user_id}.</p>"
                )
            )
            # Add email sending to background tasks
            background_tasks.add_task(send_email, email)
            print(f"Email scheduled to be sent for user: {user_id}")
        else:
            print(f"Email sending skipped for user: {user_id}")

        # Delete the .wav file if it was created
        if os.path.exists(wav_path) and wav_path != temp_file_path:
            os.remove(wav_path)
            print(f"Deleted .wav file: {wav_path}")

        return {"perceptualRating": perceptual_rating}

    except Exception as e:
        print(f"[{request_id}] OHM processing error for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Internal server error occurred during processing")


@app.post("/gop")
@limiter.limit("200/minute")
async def predict_gop(request: Request, background_tasks: BackgroundTasks, gop_request: GOPRequest, api_key: str = Depends(verify_api_key)):
    # Generate request ID for tracking
    request_id = generate_request_id()
    start_time = time.time()

    user_id = gop_request.userId
    name = gop_request.name
    community_worker_name = gop_request.communityWorkerName
    transcript = gop_request.transcript
    upload_file_name = gop_request.uploadFileName
    send_email_flag = gop_request.sendEmail

    # Validate and sanitize filename
    try:
        upload_file_name = validate_upload_filename(upload_file_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {str(e)}")

    print(f"[{request_id}] Received GOP request for userId: {user_id}")
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
                result = subprocess.run([
                    "ffmpeg", "-i", temp_file_path, wav_path
                ], capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
            else:
                wav_path = temp_file_path

        # Process the downloaded file for GOP prediction
        gop_result = compute_gop(wav_path, transcript)

        # Only send email if sendEmail is True
        if send_email_flag:
            # Prepare email data
            email = EmailSchema(
                subject="CleftCare: New GOP Assessment Result",
                body=(
                    f"<p>The patient {name} recorded by the community worker {community_worker_name}.</p>"
                    f"<p>GOP Assessment completed with sentence score: {gop_result.get('sentence_gop', 'N/A')}</p>"
                    f"<p>The link to the more details for the patient - https://cleftcare-dashboard.vercel.app/dashboard/{user_id}.</p>"
                )
            )
            # Add email sending to background tasks
            background_tasks.add_task(send_email, email)
            print(f"Email scheduled to be sent for user: {user_id}")
        else:
            print(f"Email sending skipped for user: {user_id}")

        # Delete the .wav file if it was created
        if os.path.exists(wav_path) and wav_path != temp_file_path:
            os.remove(wav_path)
            print(f"Deleted .wav file: {wav_path}")

        return gop_result

    except Exception as e:
        print(f"[{request_id}] GOP processing error for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Internal server error occurred during GOP processing")


if __name__ == "__main__":
    # Read the PORT environment variable, default to 8080 if not set
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
