<!-- Agent rule: CleftCare API integration for Expo app -->

## Objective

- Implement speech assessment integration in the Expo app using presigned S3 uploads and server-side batch processing (GOP → pick best → OHM) per sentence.
- Follow production flow by default; use dev/test flow only when explicitly requested.

## Hard rules

- Never start any dev server or background process.
- Always include `X-API-Key` when calling the FastAPI service.
- Use flat S3 object keys (no slashes). Filenames must match `^[a-zA-Z0-9._-]+$` and end with one of: `.m4a`, `.wav`, `.mp3`, `.flac`.
- Prefer batch-per-sentence processing: do not trigger ML per attempt.

## Environments

- FastAPI (ML service)
  - Base URL: `FASTAPI_URL`
  - Header: `X-API-Key: FASTAPI_API_KEY`
- Express (application backend)
  - Base URL: `API_BASE`
  - Owns AWS credentials and presigned URL generation

## Production flow (default)

1. Presign each recording attempt from the Expo app via Express:

```ts
// POST {API_BASE}/uploads/presign
// Body: { filename, contentType: 'audio/m4a', userId }
const { url, key } = await fetch(`${API_BASE}/uploads/presign`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ filename, contentType: "audio/m4a", userId }),
}).then((r) => r.json());
```

2. Upload the file directly from the device to S3 using the returned URL:

```ts
await FileSystem.uploadAsync(url, localFileUri, {
  httpMethod: "PUT",
  headers: { "Content-Type": "audio/m4a" },
  uploadType: FileSystem.FileSystemUploadType.BINARY_CONTENT,
});
// Keep the "key" for this attempt.
```

3. After the user taps Done for a sentence, submit all attempt keys for that sentence to Express:

```ts
// POST {API_BASE}/sentences/complete
// Body: { userId, name, communityWorkerName, sentenceId, transcript, language, attemptKeys }
const result = await fetch(`${API_BASE}/sentences/complete`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    userId,
    name,
    communityWorkerName,
    sentenceId,
    transcript,
    language: "kn",
    attemptKeys,
  }),
}).then((r) => r.json());
```

4. Express calls FastAPI batch endpoint to process the sentence:

- Endpoint: `POST {FASTAPI_URL}/api/v1/process-sentence`
- Headers: `Content-Type: application/json`, `X-API-Key: FASTAPI_API_KEY`
- Body shape:

```ts
type BatchProcessRequest = {
  userId: string;
  name: string;
  communityWorkerName: string;
  sentenceId: number;
  transcript: string;
  language: "kn" | "en";
  uploadFileNames: string[]; // S3 keys returned by presign (flat names)
  sendEmail: boolean; // usually false from the app flow
};
```

5. The response includes GOP for all attempts, selected best file, and OHM rating for the best file. Persist or display as needed.

```ts
type BatchProcessResponse = {
  success: boolean;
  data: {
    sentenceId: number;
    transcript: string;
    totalFiles: number;
    gopResults: Array<{
      filename: string;
      sentence_gop?: number;
      error?: string;
      perphone_gop?: any[];
      latency_ms?: number;
    }>;
    bestFile: { filename: string; gopScore: number };
    ohmRating: number | null;
    errors?: { ohm_error?: string } | null;
  };
  metadata: {
    requestId: string;
    processingTime: number;
    timestamp: number;
    error: string | null;
  };
};
```

## Dev/test flow (only when asked)

- Direct combined upload to ML service for ad hoc testing:
  - `POST {FASTAPI_URL}/api/v1/test/gop-ohm`
  - multipart/form-data fields: `wav` (WAV file), `transcript`, `language` (e.g., `kn`)
  - Header: `X-API-Key`
- Note: This endpoint expects WAV. Prefer production flow for actual app usage.

## Error handling and limits

- Respect rate limits: GOP 200/min, OHM 50/min, batch 20/min.
- Retries:
  - S3 PUT: retry on 5xx with exponential backoff.
  - FastAPI 429/5xx: backoff and retry or surface actionable error.
- Validate filenames prior to upload: ensure they remain flat and extension is allowed.

## Key naming guidance

- Generate ASCII-only, flat keys, e.g.: `<userId>_<timestamp>_<uuid>.m4a`.
- Avoid spaces and path separators. Do not use prefixes like `recordings/`.

## Security

- Expo must never hold AWS credentials.
- Only Express signs uploads; FastAPI remains ML-only.
