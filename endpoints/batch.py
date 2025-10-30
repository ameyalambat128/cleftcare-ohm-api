import time
import os
import boto3
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address

from models.schemas import BatchProcessRequest
from services.processing import AudioProcessor
from services.callback import CallbackService
from api_utils.helpers import generate_request_id, create_response, update_status, validate_upload_filename
from services.datastore import SupabaseSync


# Initialize router and limiter
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


def _format_callback_payload(
    user_id: str,
    sentence_id: int,
    transcript: str,
    language: str,
    batch_results: dict,
    request_id: str,
    processing_time_ms: int,
) -> dict:
    """Format callback payload according to integration spec."""
    # Transform perphone_gop from [(phone, score), ...] to [{"phone": phone, "score": score}, ...]
    gop_results = []
    best_gop_result = None

    # Find the best GOP result to extract per-phone scores
    for gop_result in batch_results.get("gop_results", []):
        if (
            best_gop_result is None
            or gop_result.get("sentence_gop", float("-inf"))
            > best_gop_result.get("sentence_gop", float("-inf"))
        ):
            best_gop_result = gop_result

    if best_gop_result and "perphone_gop" in best_gop_result:
        perphone_gop = best_gop_result["perphone_gop"]
        if isinstance(perphone_gop, list):
            for item in perphone_gop:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    phone_id, score = item[0], item[1]
                    gop_results.append({"phone": str(phone_id), "score": float(score)})
                elif isinstance(item, dict):
                    # Already formatted
                    gop_results.append(item)

    return {
        "userId": user_id,
        "sentenceId": str(sentence_id),
        "transcript": transcript,
        "language": language or "unknown",
        "bestFile": {
            "filename": batch_results.get("best_gop_file") or "unknown",
            "gopScore": batch_results.get("best_gop_score") or 0.0,
        },
        "ohmRating": batch_results.get("ohm_rating"),
        "gopResults": gop_results,
        "metadata": {
            "requestId": request_id,
            "processingTime": processing_time_ms,
        },
    }


def _process_and_callback(
    batch_request: BatchProcessRequest,
    callback_url: str,
    request_id: str,
    user_id: str,
    upload_file_names: list,
):
    """Background task to process audio and send callback."""
    start_time = time.time()

    try:
        # Update status to processing
        update_status(
            user_id,
            request_id,
            "processing",
            "batch-sentence",
            {
                "sentence_id": batch_request.sentenceId,
                "total_files": len(upload_file_names),
            },
        )

        # Initialize processor with S3 client
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION"),
        )
        processor = AudioProcessor(s3_client, "cleftcare-test")
        supabase_sync = SupabaseSync()

        # Process all files in the sentence
        batch_results = processor.process_sentence_batch(
            upload_file_names, batch_request.transcript, batch_request.language
        )

        # Calculate processing time
        processing_time = time.time() - start_time
        processing_time_ms = int(round(processing_time * 1000))

        # Update status to completed
        response_data = {
            "sentenceId": batch_request.sentenceId,
            "transcript": batch_request.transcript,
            "totalFiles": len(upload_file_names),
            "gopResults": batch_results["gop_results"],
            "bestFile": {
                "filename": batch_results["best_gop_file"],
                "gopScore": batch_results["best_gop_score"],
            },
            "ohmRating": batch_results.get("ohm_rating"),
            "errors": {
                "ohm_error": batch_results.get("ohm_error"),
            }
            if batch_results.get("ohm_error")
            else None,
        }
        update_status(user_id, request_id, "completed", "batch-sentence", response_data)

        # Save to Supabase
        supabase_sync.upsert_user_audio_file(
            user_id=user_id,
            prompt=batch_request.transcript,
            prompt_number=batch_request.sentenceId,
            s3_key=batch_results["best_gop_file"],
            language=batch_request.language,
            gop_sentence_score=batch_results["best_gop_score"],
            ohm_score=batch_results.get("ohm_rating"),
            all_gop_scores={
                "gopResults": batch_results["gop_results"],
            },
            per_phone_gop=None,
            request_id=request_id,
            processing_time=processing_time,
        )

        # Format and send callback
        callback_payload = _format_callback_payload(
            user_id=user_id,
            sentence_id=batch_request.sentenceId,
            transcript=batch_request.transcript,
            language=batch_request.language,
            batch_results=batch_results,
            request_id=request_id,
            processing_time_ms=processing_time_ms,
        )

        callback_service = CallbackService()
        callback_service.send_callback(callback_url, callback_payload, request_id)

    except Exception as e:
        processing_time = time.time() - start_time
        print(f"[{request_id}] Batch processing error for user {user_id}: {str(e)}")

        # Update status to failed
        update_status(
            user_id, request_id, "failed", "batch-sentence", {"error": str(e)}
        )

        # Still try to send callback with error info
        try:
            error_callback_payload = {
                "userId": user_id,
                "sentenceId": str(batch_request.sentenceId),
                "transcript": batch_request.transcript,
                "language": batch_request.language or "unknown",
                "error": str(e),
                "metadata": {
                    "requestId": request_id,
                    "processingTime": int(round(processing_time * 1000)),
                },
            }
            callback_service = CallbackService()
            callback_service.send_callback(
                callback_url, error_callback_payload, request_id
            )
        except Exception as callback_error:
            print(
                f"[{request_id}] Failed to send error callback: {str(callback_error)}"
            )


@router.post("/process-sentence")
@limiter.limit("20/minute")  # Conservative limit for batch processing
async def process_sentence_batch(
    request: Request,
    background_tasks: BackgroundTasks,
    batch_request: BatchProcessRequest,
):
    """
    Process multiple audio files for one sentence:
    1. Run GOP on all files
    2. Find the best GOP score
    3. Run OHM on the best file
    4. Return comprehensive results

    If callbackUrl is provided, returns 202 Accepted immediately and processes in background.
    Otherwise, processes synchronously and returns 200 OK with results.
    """
    # Generate request ID for tracking
    request_id = generate_request_id()
    start_time = time.time()

    user_id = batch_request.userId
    sentence_id = batch_request.sentenceId
    transcript = batch_request.transcript
    upload_file_names = batch_request.uploadFileNames
    send_email_flag = batch_request.sendEmail
    callback_url = batch_request.callbackUrl

    print(
        f"[{request_id}] Received batch processing request for userId: {user_id}, "
        f"sentence: {sentence_id}, callbackUrl: {callback_url or 'none'}"
    )

    # Validate all filenames
    try:
        validated_filenames = []
        for filename in upload_file_names:
            validated_filenames.append(validate_upload_filename(filename))
        upload_file_names = validated_filenames
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {str(e)}")

    # If callbackUrl is provided, process asynchronously
    if callback_url:
        # Add background task for processing
        background_tasks.add_task(
            _process_and_callback,
            batch_request,
            callback_url,
            request_id,
            user_id,
            upload_file_names,
        )

        # Return 202 Accepted immediately
        return Response(
            content='{"message": "Processing started. Results will be sent to callback URL."}',
            status_code=202,
            media_type="application/json",
        )

    # Synchronous processing (backward compatibility - no callbackUrl)
    try:
        # Initialize processor with S3 client
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION"),
        )
        processor = AudioProcessor(s3_client, "cleftcare-test")
        supabase_sync = SupabaseSync()

        # Update status to processing
        update_status(
            user_id,
            request_id,
            "processing",
            "batch-sentence",
            {
                "sentence_id": sentence_id,
                "total_files": len(upload_file_names),
            },
        )

        # Process all files in the sentence
        batch_results = processor.process_sentence_batch(
            upload_file_names, transcript, batch_request.language
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
                "gopScore": batch_results["best_gop_score"],
            },
            "ohmRating": batch_results.get("ohm_rating"),
            "errors": {
                "ohm_error": batch_results.get("ohm_error"),
            }
            if batch_results.get("ohm_error")
            else None,
        }

        # Handle email notification if requested
        if send_email_flag:
            print(f"[{request_id}] Email notification requested for batch processing")

        # Update status to completed
        update_status(user_id, request_id, "completed", "batch-sentence", response_data)

        supabase_sync.upsert_user_audio_file(
            user_id=user_id,
            prompt=batch_request.transcript,
            prompt_number=sentence_id,
            s3_key=batch_results["best_gop_file"],
            language=batch_request.language,
            gop_sentence_score=batch_results["best_gop_score"],
            ohm_score=batch_results.get("ohm_rating"),
            all_gop_scores={
                "gopResults": batch_results["gop_results"],
            },
            per_phone_gop=None,
            request_id=request_id,
            processing_time=processing_time,
        )

        return create_response(
            success=True,
            data=response_data,
            request_id=request_id,
            processing_time=processing_time,
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
            error_message="Internal server error occurred during batch processing",
        )
        raise HTTPException(status_code=500, detail=error_response)