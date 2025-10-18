# CleftCare API Integration (Expo + Express + FastAPI)

This guide shows how to integrate the speech assessment API with an Expo app using S3 presigned uploads via an Express backend, and ML processing via FastAPI.

## Overview

- Expo uploads each audio attempt directly to S3 using a presigned URL from Express.
- When the user taps Done for a sentence, Express calls FastAPI to run GOP on all attempts, selects the best, and runs OHM on the best file.
- FastAPI returns combined results; Express can persist them to a database.

## Endpoints

- Express
  - POST `/uploads/presign` → `{ url, key, expiresIn }`
  - POST `/sentences/complete` → triggers ML batch processing
- FastAPI (ML)
  - POST `/api/v1/process-sentence` → GOP+OHM for one sentence
  - (Dev) POST `/api/v1/test/gop-ohm` → combined test with WAV upload

## Key rules

- Filenames must be flat (no slashes) with allowed characters `^[a-zA-Z0-9._-]+$` and extension in {`.m4a`, `.wav`, `.mp3`, `.flac`}.
- Suggested key format: `<userId>_<timestamp>_<uuid>.m4a`.
- Always include `X-API-Key` when calling FastAPI.

## Expo: Upload flow

```ts
// 1) Presign
const { url, key } = await fetch(`${API_BASE}/uploads/presign`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ filename, contentType: "audio/m4a", userId }),
}).then((r) => r.json());

// 2) Upload to S3
await FileSystem.uploadAsync(url, localFileUri, {
  httpMethod: "PUT",
  headers: { "Content-Type": "audio/m4a" },
  uploadType: FileSystem.FileSystemUploadType.BINARY_CONTENT,
});

// Save `key` to include in the sentence submission
```

## Expo: Submit sentence for processing

```ts
await fetch(`${API_BASE}/sentences/complete`, {
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
});
```

## Express: Presign route (TypeScript)

```ts
import { Router } from "express";
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import crypto from "crypto";
const r = Router();
const s3 = new S3Client({ region: process.env.AWS_REGION });
const sanitize = (s: string) => s.replace(/[^a-zA-Z0-9._-]/g, "-");

r.post("/uploads/presign", async (req, res) => {
  const { filename, contentType, userId } = req.body;
  if (!filename || !contentType || !userId)
    return res.status(400).json({ error: "Bad request" });
  const safeName = sanitize(filename);
  const ext = safeName.includes(".") ? safeName.split(".").pop() : "m4a";
  const key = `${sanitize(userId)}_${Date.now()}_${crypto.randomUUID()}.${ext}`;
  const url = await getSignedUrl(
    s3,
    new PutObjectCommand({
      Bucket: process.env.S3_BUCKET!,
      Key: key,
      ContentType: contentType,
    }),
    { expiresIn: 300 }
  );
  res.json({ url, key, expiresIn: 300 });
});

export default r;
```

## Express: Sentence completion route

```ts
r.post("/sentences/complete", async (req, res) => {
  const {
    userId,
    name,
    communityWorkerName,
    sentenceId,
    transcript,
    language,
    attemptKeys,
  } = req.body;
  if (
    !userId ||
    !sentenceId ||
    !transcript ||
    !Array.isArray(attemptKeys) ||
    attemptKeys.length === 0
  ) {
    return res.status(400).json({ error: "Missing fields" });
  }
  // Optional: persist attempts in your DB
  const fastapiRes = await fetch(
    `${process.env.FASTAPI_URL}/api/v1/process-sentence`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": process.env.FASTAPI_API_KEY!,
      },
      body: JSON.stringify({
        userId,
        name,
        communityWorkerName,
        sentenceId,
        transcript,
        language,
        uploadFileNames: attemptKeys,
        sendEmail: false,
      }),
    }
  );
  const data = await fastapiRes.json();
  res.status(fastapiRes.status).json(data);
});
```

## FastAPI: Expected request body

```ts
type BatchProcessRequest = {
  userId: string;
  name: string;
  communityWorkerName: string;
  sentenceId: number;
  transcript: string;
  language: "kn" | "en";
  uploadFileNames: string[];
  sendEmail: boolean;
};
```

## S3 CORS example

```json
[
  {
    "AllowedHeaders": ["*"],
    "AllowedMethods": ["PUT", "GET", "HEAD"],
    "AllowedOrigins": [
      "https://your-app.example.com",
      "http://localhost:19006"
    ],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3000
  }
]
```

## IAM policy example (Express principal)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject"],
      "Resource": "arn:aws:s3:::YOUR_BUCKET/*"
    }
  ]
}
```

## Troubleshooting

- 403 on PUT: wrong Content-Type header or URL expired; re-presign.
- 415/400 at FastAPI: ensure filenames are flat and extensions allowed.
- 429 at FastAPI: backoff and retry; avoid concurrent batch calls.
