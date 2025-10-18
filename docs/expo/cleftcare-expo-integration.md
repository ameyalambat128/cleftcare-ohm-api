### CleftCare Expo Integration (Presigned S3 + Batch ML)

#### What this does

- Record multiple attempts for a sentence in Expo
- Upload each attempt directly to S3 via Express presigned URLs
- On Done (per sentence): send all attempt keys to Express → Express calls FastAPI `/api/v1/process-sentence` → returns GOP for all, best file, and OHM for the best

#### Envs (Expo app)

- API_BASE: base URL of your Express backend (e.g., `https://api.example.com`)
- Do not call FastAPI directly in production from Expo

#### Types

```ts
export type PresignResponse = { url: string; key: string; expiresIn: number };

export type SentenceCompleteRequest = {
  userId: string;
  name: string;
  communityWorkerName: string;
  sentenceId: number;
  transcript: string;
  language: "kn" | "en";
  attemptKeys: string[];
};

export type BatchResult = {
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

#### Helpers (Expo)

```ts
const jsonFetch = async <T>(url: string, init?: RequestInit) => {
  const res = await fetch(url, init);
  const body = await res.json().catch(() => ({}));
  if (!res.ok)
    throw Object.assign(new Error("HTTP " + res.status), {
      status: res.status,
      body,
    });
  return body as T;
};

export const presignAttempt = async (
  apiBase: string,
  filename: string,
  contentType: string,
  userId: string
) => {
  return jsonFetch<PresignResponse>(`${apiBase}/uploads/presign`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename, contentType, userId }),
  });
};

export const uploadAttemptToS3 = async (
  signedUrl: string,
  localFileUri: string,
  contentType: string
) => {
  // expo-file-system
  const { default: FileSystem } = await import("expo-file-system");
  const result = await FileSystem.uploadAsync(signedUrl, localFileUri, {
    httpMethod: "PUT",
    headers: { "Content-Type": contentType },
    uploadType: FileSystem.FileSystemUploadType.BINARY_CONTENT,
  });
  if (result.status !== 200 && result.status !== 204) {
    throw new Error(`S3 upload failed: ${result.status}`);
  }
};

export const completeSentence = async (
  apiBase: string,
  req: SentenceCompleteRequest
) => {
  return jsonFetch<BatchResult>(`${apiBase}/sentences/complete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
};
```

#### Example usage (screen logic)

```ts
// Pseudocode: per sentence flow
const attemptKeys: string[] = [];

async function onRecordAttemptDone(localUri: string) {
  const filename = "attempt.m4a"; // name for presign; server will return a flat key
  const contentType = "audio/m4a";
  const { url, key } = await presignAttempt(
    API_BASE,
    filename,
    contentType,
    userId
  );
  await uploadAttemptToS3(url, localUri, contentType);
  attemptKeys.push(key);
}

async function onSentenceDone() {
  if (attemptKeys.length === 0) return;
  const result = await completeSentence(API_BASE, {
    userId,
    name,
    communityWorkerName,
    sentenceId,
    transcript,
    language: "kn",
    attemptKeys,
  });
  // result.data.bestFile, result.data.ohmRating, etc.
}
```

#### Dev test (optional)

- For ad hoc testing against FastAPI directly (not production flow):
  - `POST /api/v1/test/gop-ohm` (multipart/form-data) with WAV + `transcript` + `language`
  - Header: `X-API-Key`

#### Constraints & tips

- Keys returned by presign are flat strings (no `/`), ASCII `[a-zA-Z0-9._-]`, extensions in {`.m4a`, `.wav`, `.mp3`, `.flac`}
- Keep `Content-Type` consistent between presign request and S3 PUT
- Backoff on network errors and 5xx; re-presign if you hit 403/SignatureDoesNotMatch

#### Agent rules (to copy into Expo repo .cursor/rules)

- Do not start dev servers
- Use Express endpoints only in production flow
- Use `/uploads/presign` → PUT to S3 → collect `key` per attempt
- On Done: call `/sentences/complete` with all `attemptKeys`
- Do not call FastAPI directly from the app except in explicit dev-test tasks
- Ensure `Content-Type` header matches the presigned request
