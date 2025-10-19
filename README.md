# Cleft Care OHM API

Speech assessment API combining OHM (Oral Hypernasality Measure) and GOP (Goodness of Pronunciation) models for cleft lip/palate and Kannada speech quality assessment.

## Quick Start

### Development Mode (Recommended for Testing)

1. **Setup environment:**

   ```bash
   cp .env.example .env
   # Edit .env if needed - defaults work for development
   ```

2. **Add your audio files:**

   ```bash
   # Place test audio files in audios/ or audios/samples/ directory
   # In development, the API looks in audios/ first, then audios/samples/
   cp your-audio-files.m4a audios/samples/
   ```

3. **Start the API:**

   ```bash
   docker-compose --profile dev up
   ```

4. **Test the endpoints:**

   ```bash
   # Health check
   curl http://localhost:8080/

   # OHM assessment
   curl -X POST http://localhost:8080/ohm \
     -H "Content-Type: application/json" \
     -H "X-API-Key: dev-key-12345" \
     -d '{"userId": "test", "name": "Test", "communityWorkerName": "Test", "promptNumber": 1, "language": "kn", "uploadFileName": "your-file.m4a", "sendEmail": false}'
   ```

### Production Mode

1. **Deploy:**
   ```bash
   docker buildx build --platform linux/amd64 -t cleftcare-ohm-api .
   ```

## API Endpoints

All endpoints require `X-API-Key` header for authentication.

### Health Check

```bash
GET /
```

Returns API status and version info.

---

### OHM - Hypernasality Assessment

```bash
POST /ohm
```

Measures oral hypernasality in speech (rating 1-5, higher = more hypernasal).

**Request:**

```json
{
  "userId": "user123",
  "name": "Patient Name",
  "communityWorkerName": "Worker Name",
  "promptNumber": 1,
  "language": "kn",
  "uploadFileName": "audio.m4a",
  "sendEmail": false
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "rating": 2.4356,
    "userId": "user123",
    "name": "Patient Name",
    "communityWorkerName": "Worker Name",
    "promptNumber": 1,
    "language": "kn"
  },
  "metadata": {
    "requestId": "550e8400-e29b-41d4-a716-446655440000",
    "processingTime": 1.234,
    "timestamp": 1738176080542,
    "error": null
  }
}
```

**Rate limit:** 50 requests/minute

---

### GOP - Pronunciation Assessment

```bash
POST /gop
```

Measures pronunciation quality (higher/less negative scores = better pronunciation).

**Request:**

```json
{
  "userId": "user123",
  "transcript": "ಪಟ್ಟಿ",
  "uploadFileName": "audio.m4a"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "sentence_gop": -1.0301829,
    "perphone_gop": [
      { "phone": "p", "score": -0.523 },
      { "phone": "a", "score": -1.234 }
    ],
    "latency_ms": 850
  },
  "metadata": {
    "requestId": "550e8400-e29b-41d4-a716-446655440001",
    "processingTime": 0.85,
    "timestamp": 1738176081392,
    "error": null
  }
}
```

**Rate limit:** 200 requests/minute

---

### Batch Processing - Complete Sentence Assessment

```bash
POST /api/v1/process-sentence
```

Processes multiple audio attempts for one sentence:

1. Runs GOP on all attempts
2. Selects best pronunciation (highest GOP score)
3. Runs OHM on best audio
4. Returns comprehensive results

**Request:**

```json
{
  "userId": "user123",
  "name": "Patient Name",
  "communityWorkerName": "Worker Name",
  "transcript": "ಪಟ್ಟಿ",
  "language": "kn",
  "uploadFileNames": ["attempt1.m4a", "attempt2.m4a", "attempt3.m4a"],
  "sendEmail": false
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "best_gop_score": -0.868,
    "best_file": "attempt2.m4a",
    "ohm_rating": 1.4418,
    "all_gop_scores": {
      "attempt1.m4a": -1.234,
      "attempt2.m4a": -0.868,
      "attempt3.m4a": -1.567
    },
    "userId": "user123",
    "name": "Patient Name",
    "transcript": "ಪಟ್ಟಿ",
    "language": "kn"
  },
  "metadata": {
    "requestId": "550e8400-e29b-41d4-a716-446655440002",
    "processingTime": 45.678,
    "timestamp": 1738176127070,
    "error": null
  }
}
```

**Typical workflow:** User records 2-5 attempts per sentence → API finds best pronunciation → Returns GOP + OHM scores
**Processing time:** ~30-60 seconds per sentence (depending on number of attempts)

---

### File Upload Endpoints (Development/Testing)

```bash
POST /api/v1/gop/upload
```

Direct file upload for GOP testing (multipart/form-data).

```bash
POST /api/v1/test/gop-ohm
```

Direct file upload for combined GOP+OHM testing (multipart/form-data).

**Example:**

```bash
curl -X POST http://localhost:8080/api/v1/test/gop-ohm \
  -H "X-API-Key: dev-key-12345" \
  -F "wav=@audio.wav" \
  -F "transcript=ಪಟ್ಟಿ" \
  -F "language=kn"
```

---

### Response Format

All endpoints return standardized responses:

```json
{
  "success": true,           // Request succeeded
  "data": { ... },          // Endpoint-specific results
  "metadata": {
    "requestId": "uuid",    // Unique request identifier
    "processingTime": 2.45, // Seconds
    "timestamp": 1738176080542,
    "error": null           // Error message if failed
  }
}
```

## Development vs Production

### Development Mode

- **Audio files**: Local files in `audios/` directory (falls back to `audios/samples/`)
- **Authentication**: Simple API key (`dev-key-12345`)
- **Setup**: No AWS credentials needed
- **Usage**: `docker-compose --profile dev up`

### Production Mode

- **Audio files**: Downloaded from AWS S3
- **Authentication**: Secure API key via environment
- **Setup**: Requires AWS credentials and S3 bucket
- **Usage**: `docker-compose up cleftcare-api`

## Environment Variables

| Variable                    | Description               | Development     | Production   |
| --------------------------- | ------------------------- | --------------- | ------------ |
| `ENVIRONMENT`               | Mode setting              | `development`   | `production` |
| `API_KEY`                   | Authentication key        | `dev-key-12345` | `secure-key` |
| `AWS_ACCESS_KEY_ID`         | S3 access key             | Not needed      | Required     |
| `AWS_SECRET_ACCESS_KEY`     | S3 secret key             | Not needed      | Required     |
| `AWS_DEFAULT_REGION`        | S3 region                 | Not needed      | Required     |
| `SUPABASE_URL`              | Supabase project URL      | Optional        | Required     |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key | Optional        | Required     |

## Docker Commands

### Development

```bash
# Quick start for testing
docker-compose --profile dev up

# Test Kaldi setup
docker-compose --profile test up test
```

### Production

```bash
# Local production testing
docker-compose up cleftcare-api

# Cloud Run deployment
docker buildx build --platform linux/amd64 -t cleftcare-ohm-api .
docker tag cleftcare-ohm-api us-east1-docker.pkg.dev/cleftcare/cleftcare-ohm/cleftcare-ohm:latest
docker push us-east1-docker.pkg.dev/cleftcare/cleftcare-ohm/cleftcare-ohm:latest
```

## Architecture

```
├── app.py              # Main FastAPI application
├── audios/             # Local audio files (development)
├── endpoints/          # API route handlers
├── models/             # Pydantic schemas + ML models
├── services/           # Business logic & processing
└── utils/              # Helper functions
```

## Tech Stack

- **Framework**: FastAPI with authentication & rate limiting
- **ML Models**: PyTorch (OHM), Kaldi (GOP)
- **Audio**: FFmpeg conversion, librosa processing
- **Storage**: AWS S3 (production) or local files (development)
- **Container**: Docker with multi-stage build for Kaldi integration
