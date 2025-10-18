# OHM Integration

## Summary

Front-end clients submit sentence recordings to the CleftCare API. The API forwards each request directly to the OHM service and immediately returns `202 Accepted`. OHM is responsible for all long-running processing, persistence, and notifications.

## Request Path

1. Client calls `POST /api/sentences/complete` (CleftCare API).
2. CleftCare API validates the payload and forwards it to OHM.
3. API returns `202` with a brief message. No polling endpoint or queued job is created on the Express side.
4. OHM processes audio, updates its own datastore, and—if needed—writes back to shared tables or triggers downstream actions.

## Payload Shape

```json
{
  "userId": "string",
  "name": "optional string",
  "communityWorkerName": "optional string",
  "sentenceId": "string",
  "transcript": "string",
  "language": "optional string",
  "uploadFileNames": ["s3/object/key.m4a"],
  "sendEmail": false
}
```

### Notes

- `sentenceId` is coerced to a string before forwarding.
- `uploadFileNames` contains the S3 keys returned by the presign route. Files already exist in S3.
- `sendEmail` is currently always `false`; OHM may ignore it or use it for future features.
- If any required field is missing, the API does **not** contact OHM.

## HTTP Details

- Endpoint invoked by CleftCare API: `POST ${CLEFTCARE_OHM_URL}/api/v1/process-sentence`
- Headers include: `Content-Type: application/json`, `X-API-Key: ${CLEFTCARE_OHM_API_KEY}`
- The API does not retry or wait for a response. Errors are logged server-side only.

## OHM Responsibilities

- Handle long-running processing (GOP/OHM scoring, analytics, etc.).
- Persist results and metadata in OHM-controlled storage (database, cache, etc.).
- Update shared tables or notify clients as needed (email, push, callbacks).
- Manage retries, circuit-breaking, and monitoring around the ML pipeline.

## Operational Expectations

- OHM must return an HTTP status quickly (ideally < 2s). The CleftCare API ignores the body but logs non-2xx responses.
- Any failure handling (retries, compensating actions) is owned by OHM.
- If OHM updates CleftCare’s Postgres tables, do so using its own worker processes or APIs.

## Developer Checklist

- Ensure `CLEFTCARE_OHM_URL` and `CLEFTCARE_OHM_API_KEY` are configured for both systems.
- Monitor OHM logs for incoming requests and processing errors.
- Coordinate schema changes if OHM writes back to shared tables.
- Document any additional signals (webhooks, events) that other teams should consume.
