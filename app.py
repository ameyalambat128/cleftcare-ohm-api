from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Request, UploadFile, File, Form
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

# Import new modular components
from models.schemas import PredictRequest, GOPRequest, EmailSchema, APIResponse
from services.processing import AudioProcessor
from api_utils.helpers import generate_request_id, validate_upload_filename, create_response, update_status, status_tracking
from endpoints.batch import router as batch_router

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

# Initialize AudioProcessor for dependency injection
audio_processor = None

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

# S3 Client Setup
S3_BUCKET_NAME = 'cleftcare-test'
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)

# Initialize global AudioProcessor after S3 client is ready
def get_audio_processor():
    global audio_processor
    if audio_processor is None:
        audio_processor = AudioProcessor(s3, S3_BUCKET_NAME)
    return audio_processor


"""
Lightweight test endpoints (development): run OHM or GOP individually on a given filename.
- In development: reads from audios/ or audios/samples/
- In production: downloads from S3
"""

def _sanitize_floats_for_json(obj):
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats_for_json(elem) for elem in obj]
    return obj

class TestOHMInput(BaseModel):
    uploadFileName: str
    language: str


@app.post("/api/v1/test/ohm")
async def test_ohm(body: TestOHMInput):
    processor = get_audio_processor()
    try:
        wav_path = processor.download_and_convert_audio(body.uploadFileName)
        rating = predict_ohm_rating(Path(wav_path).parent, body.language)
        return _sanitize_floats_for_json({"success": True, "ohmRating": rating, "wavPath": wav_path})
    except Exception as e:
        return {"success": False, "error": str(e)}


class TestGOPInput(BaseModel):
    uploadFileName: str
    transcript: str


@app.post("/api/v1/test/gop")
async def test_gop(body: TestGOPInput):
    processor = get_audio_processor()
    try:
        wav_path = processor.download_and_convert_audio(body.uploadFileName)
        result = compute_gop(wav_path, body.transcript)
        return _sanitize_floats_for_json({"success": True, "gop": result, "wavPath": wav_path})
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/v1/gop/upload")
async def gop_upload(
    wav: UploadFile = File(...),
    transcript: str = Form(...)
):
    """
    Local file upload endpoint for GOP testing.
    Similar to shruthi-gop-original Flask endpoint.
    Accepts: multipart/form-data with 'wav' file and 'transcript' text
    """
    import tempfile

    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            content = await wav.read()
            tmp_wav.write(content)
            wav_path = tmp_wav.name

        # Process with GOP
        result = compute_gop(wav_path, transcript)

        # Clean up temporary file
        os.remove(wav_path)

        return _sanitize_floats_for_json(result)

    except Exception as e:
        # Clean up on error
        if 'wav_path' in locals() and os.path.exists(wav_path):
            os.remove(wav_path)
        return {"error": str(e), "utt_id": None, "sentence_gop": None}


@app.post("/api/v1/test/gop-ohm")
async def test_gop_ohm(
    wav: UploadFile = File(...),
    transcript: str = Form(...),
    language: str = Form(default="kn")
):
    """
    Test endpoint for GOP+OHM processing with file upload.
    Processes audio with both GOP and OHM models.

    Args:
        wav: Audio file (WAV format preferred)
        transcript: Expected transcript for GOP scoring
        language: Language code (default: 'kn' for Kannada)

    Returns:
        Combined GOP and OHM results
    """
    import tempfile
    from pathlib import Path

    wav_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp") as tmp_wav:
            content = await wav.read()
            tmp_wav.write(content)
            wav_path = tmp_wav.name

        # Process with GOP
        gop_result = compute_gop(wav_path, transcript)

        # Process with OHM
        temp_folder = Path(wav_path).parent
        ohm_rating = predict_ohm_rating(temp_folder, language)

        # Combine results
        combined_result = {
            "gop": {
                "sentence_gop": gop_result.get("sentence_gop"),
                "perphone_gop": gop_result.get("perphone_gop", []),
                "latency_ms": gop_result.get("latency_ms"),
                "error": gop_result.get("error")
            },
            "ohm": {
                "rating": ohm_rating
            },
            "input": {
                "transcript": transcript,
                "language": language,
                "filename": wav.filename
            }
        }

        # Clean up temporary file
        os.remove(wav_path)

        return _sanitize_floats_for_json(combined_result)

    except Exception as e:
        # Clean up on error
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
        return {
            "error": str(e),
            "gop": None,
            "ohm": None
        }


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
Existing Endpoints - Keep all functionality intact
"""

@app.get("/status/{user_id}")
async def get_status(user_id: str, api_key: str = Depends(verify_api_key)):
    """Get processing status for a user"""
    if user_id not in status_tracking:
        return create_response(
            success=True,
            data={"requests": []},
            request_id="status-check",
            processing_time=0.001
        )

    # Get recent requests (last 100)
    recent_requests = status_tracking[user_id][-100:]

    return create_response(
        success=True,
        data={"requests": recent_requests},
        request_id="status-check",
        processing_time=0.001
    )


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

    # Update status to processing
    update_status(user_id, request_id, "processing", "ohm")
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

        # Calculate processing time and create response
        processing_time = time.time() - start_time
        response_data = {"perceptualRating": perceptual_rating}

        # Update status to completed
        update_status(user_id, request_id, "completed", "ohm", response_data)

        return create_response(
            success=True,
            data=response_data,
            request_id=request_id,
            processing_time=processing_time
        )

    except Exception as e:
        processing_time = time.time() - start_time
        print(f"[{request_id}] OHM processing error for user {user_id}: {str(e)}")

        # Update status to failed
        update_status(user_id, request_id, "failed", "ohm", {"error": str(e)})

        error_response = create_response(
            success=False,
            data={},
            request_id=request_id,
            processing_time=processing_time,
            error_message="Internal server error occurred during processing"
        )
        raise HTTPException(status_code=500, detail=error_response)


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

    # Update status to processing
    update_status(user_id, request_id, "processing", "gop")
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

        # Calculate processing time and create response
        processing_time = time.time() - start_time

        # Update status to completed
        update_status(user_id, request_id, "completed", "gop", gop_result)

        return create_response(
            success=True,
            data=gop_result,
            request_id=request_id,
            processing_time=processing_time
        )

    except Exception as e:
        processing_time = time.time() - start_time
        print(f"[{request_id}] GOP processing error for user {user_id}: {str(e)}")

        # Update status to failed
        update_status(user_id, request_id, "failed", "gop", {"error": str(e)})

        error_response = create_response(
            success=False,
            data={},
            request_id=request_id,
            processing_time=processing_time,
            error_message="Internal server error occurred during GOP processing"
        )
        raise HTTPException(status_code=500, detail=error_response)


# Add new batch processing endpoints with API key requirement
app.include_router(
    batch_router,
    prefix="/api/v1",
    dependencies=[Depends(verify_api_key)]
)


if __name__ == "__main__":
    # Read the PORT environment variable, default to 8080 if not set
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
