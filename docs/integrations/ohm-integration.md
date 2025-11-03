# OHM Integration

## Summary

Front-end clients submit sentence recordings to the CleftCare API. The API forwards each request directly to the OHM service and immediately returns `202 Accepted`. OHM is responsible for all long-running processing, persistence, and notifications. When processing completes, OHM calls back to the CleftCare API to save results and trigger automatic average score calculations.

## Request Path

1. Client calls `POST /api/sentences/complete` (CleftCare API).
2. CleftCare API validates the payload and forwards it to OHM with a `callbackUrl`.
3. API returns `202` with a brief message. No polling endpoint or queued job is created on the Express side.
4. OHM processes audio (long-running operation).
5. **OHM calls the `callbackUrl` when processing completes** with results (GOP scores, OHM scores, audio file metadata).
6. CleftCare API callback endpoint saves `UserAudioFile` records and automatically calculates/updates average scores for the user.

## Payload Shape (CleftCare → OHM)

```json
{
  "userId": "string",
  "name": "optional string",
  "communityWorkerName": "optional string",
  "sentenceId": "string",
  "transcript": "string",
  "language": "optional string",
  "uploadFileNames": ["s3/object/key.m4a"],
  "sendEmail": false,
  "callbackUrl": "https://your-api.com/api/ohm-callback"
}
```

### Notes

- `sentenceId` is coerced to a string before forwarding.
- `uploadFileNames` contains the S3 keys returned by the presign route. Files already exist in S3.
- `sendEmail` is currently always `false`; OHM may ignore it or use it for future features.
- `callbackUrl` is the endpoint OHM should POST results to when processing completes.
- If any required field is missing, the API does **not** contact OHM.

## Callback Payload Shape (OHM → CleftCare)

When OHM completes processing, it POSTs results to the provided `callbackUrl`:

```json
{
  "userId": "string",
  "sentenceId": "string",
  "transcript": "string",
  "language": "string",
  "bestFile": {
    "filename": "s3/key/to/best/audio.m4a",
    "gopScore": 85.5
  },
  "ohmRating": 4.2,
  "gopResults": [
    {
      "phone": "AH",
      "score": 90.2
    },
    {
      "phone": "K",
      "score": 78.5
    }
  ],
  "metadata": {
    "requestId": "uuid",
    "processingTime": 12500
  }
}
```

### Callback Notes

- `bestFile.filename`: S3 key of the selected audio file from the attempts
- `bestFile.gopScore`: Sentence-level GOP score (averaged across phones)
- `ohmRating`: Overall OHM intelligibility score
- `gopResults`: Per-phone GOP scores for detailed analysis
- `metadata.processingTime`: Time in milliseconds OHM took to process
- CleftCare API validates this payload and saves it as a `UserAudioFile` record

## HTTP Details

- Endpoint invoked by CleftCare API: `POST ${CLEFTCARE_OHM_URL}/api/v1/process-sentence`
- Headers include: `Content-Type: application/json`, `X-API-Key: ${CLEFTCARE_OHM_API_KEY}`
- The API does not retry or wait for a response. Errors are logged server-side only.

## OHM Responsibilities

- Handle long-running processing (GOP/OHM scoring, analytics, etc.).
- Call the provided `callbackUrl` when processing completes with results.
- Implement retry logic for callback requests (exponential backoff recommended).
- Manage circuit-breaking and monitoring around the ML pipeline.
- Log callback failures for debugging and alerting.

## CleftCare API Responsibilities

- Provide a stable, publicly accessible callback endpoint (`POST /api/ohm-callback`).
- Validate incoming callback payloads from OHM.
- Save `UserAudioFile` records with GOP and OHM scores.
- Automatically trigger average score calculations for the user.
- Handle idempotency (same callback might arrive multiple times due to retries).
- Log callback processing for monitoring and debugging.

## Operational Expectations

- OHM must return an HTTP status quickly (ideally < 2s) to the initial `/process-sentence` request. The CleftCare API ignores the body but logs non-2xx responses.
- OHM should call the callback endpoint within a reasonable time after processing (typically < 5 minutes, but depends on queue depth).
- Callback endpoint should respond with `200 OK` on success or `4xx/5xx` on validation/processing errors.
- Any failure handling (retries, compensating actions) for the callback is owned by OHM.
- CleftCare API must ensure callback endpoint is protected but accessible to OHM (consider API key authentication or IP whitelisting).

## Developer Checklist

### For CleftCare API Team:
- Implement `POST /api/ohm-callback` endpoint to receive results from OHM.
- Update `/api/sentences/complete` to include `callbackUrl` in payload to OHM.
- Set `CALLBACK_BASE_URL` environment variable (e.g., `https://api.cleftcare.com`).
- Ensure callback endpoint is publicly accessible (not behind authentication for OHM requests).
- Consider adding a shared secret or signature validation for callback security.
- Test callback handling with mock OHM responses.
- Monitor callback endpoint for errors and failed processing.

### For OHM API Team:
- Update `/api/v1/process-sentence` to accept `callbackUrl` parameter.
- Implement callback mechanism to POST results to `callbackUrl` when processing completes.
- Implement retry logic with exponential backoff (3-5 retries recommended).
- Log callback attempts and failures for debugging.
- Ensure callback includes all required fields (see "Callback Payload Shape" above).
- Test callback with CleftCare API staging environment.

### Shared:
- Ensure `CLEFTCARE_OHM_URL` and `CLEFTCARE_OHM_API_KEY` are configured for both systems.
- Coordinate schema changes if data structure evolves.
- Set up monitoring and alerting for callback failures on both sides.
