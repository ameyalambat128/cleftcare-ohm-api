import time
import os
import boto3
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from models.schemas import BatchProcessRequest
from services.processing import AudioProcessor
from api_utils.helpers import generate_request_id, create_response, update_status, validate_upload_filename


# Initialize router and limiter
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/process-sentence")
@limiter.limit("20/minute")  # Conservative limit for batch processing
async def process_sentence_batch(
    request: Request,
    background_tasks: BackgroundTasks,
    batch_request: BatchProcessRequest
):
    """
    Process multiple audio files for one sentence:
    1. Run GOP on all files
    2. Find the best GOP score
    3. Run OHM on the best file
    4. Return comprehensive results
    """
    # Generate request ID for tracking
    request_id = generate_request_id()
    start_time = time.time()

    user_id = batch_request.userId
    sentence_id = batch_request.sentenceId
    transcript = batch_request.transcript
    upload_file_names = batch_request.uploadFileNames
    send_email_flag = batch_request.sendEmail

    print(f"[{request_id}] Received batch processing request for userId: {user_id}, sentence: {sentence_id}")

    # Validate all filenames
    try:
        validated_filenames = []
        for filename in upload_file_names:
            validated_filenames.append(validate_upload_filename(filename))
        upload_file_names = validated_filenames
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {str(e)}")

    # Update status to processing
    update_status(user_id, request_id, "processing", "batch-sentence", {
        "sentence_id": sentence_id,
        "total_files": len(upload_file_names)
    })

    try:
        # Initialize processor with S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION')
        )
        processor = AudioProcessor(s3_client, 'cleftcare-test')

        # Process all files in the sentence
        batch_results = processor.process_sentence_batch(
            upload_file_names,
            transcript,
            batch_request.language
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Prepare response data
        response_data = {
            "sentenceId": sentence_id,
            "transcript": transcript,
            "totalFiles": len(upload_file_names),
            "gopResults": batch_results["gop_results"],
            "bestFile": {
                "filename": batch_results["best_gop_file"],
                "gopScore": batch_results["best_gop_score"]
            },
            "ohmRating": batch_results.get("ohm_rating"),
            "errors": {
                "ohm_error": batch_results.get("ohm_error")
            } if batch_results.get("ohm_error") else None
        }

        # Handle email notification if requested
        if send_email_flag:
            # Add email to background tasks (will need to import email functionality)
            print(f"[{request_id}] Email notification requested for batch processing")

        # Update status to completed
        update_status(user_id, request_id, "completed", "batch-sentence", response_data)

        return create_response(
            success=True,
            data=response_data,
            request_id=request_id,
            processing_time=processing_time
        )

    except Exception as e:
        processing_time = time.time() - start_time
        print(f"[{request_id}] Batch processing error for user {user_id}: {str(e)}")

        # Update status to failed
        update_status(user_id, request_id, "failed", "batch-sentence", {"error": str(e)})

        error_response = create_response(
            success=False,
            data={},
            request_id=request_id,
            processing_time=processing_time,
            error_message="Internal server error occurred during batch processing"
        )
        raise HTTPException(status_code=500, detail=error_response)