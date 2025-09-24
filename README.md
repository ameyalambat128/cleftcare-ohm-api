# Cleft Care OHM API

## Tech Stack

- Web Framework: FastAPI
- Machine Learning: PyTorch, scikit-learn, transformers
- Data Processing: NumPy, pandas, librosa
- AWS Integration: S3 with Boto3
- Audio Processing: audioread, soundfile

## Endpoints

**Health Check**

- URL: `/`
- Method: `GET`
- Description: Checks the health status of the API.

**Predict OHM Rating (Test)**

- URL: `/predict-test`
- Method: `GET`
- Description: Downloads a test audio file from S3, processes it, and returns the perceptual rating.

**OHM Assessment**

- URL: `/ohm`
- Method: `POST`
- Description: Accepts a POST request with user details and audio file info, downloads the specified audio file from S3, processes it with OHM model, and returns the perceptual rating.

**Request Body**

```json
{
  "userId": "string",
  "name": "string",
  "communityWorkerName": "string",
  "promptNumber": 1,
  "language": "en",
  "uploadFileName": "string",
  "sendEmail": true
}
```

**Response**

```json
{
  "perceptualRating": <rating>
}
```

**GOP Assessment**

- URL: `/gop`
- Method: `POST`
- Description: Accepts a POST request with audio file info and transcript, downloads the specified audio file from S3, processes it with Kaldi GOP system, and returns pronunciation assessment scores.

**Request Body**

```json
{
  "userId": "string",
  "name": "string",
  "communityWorkerName": "string",
  "transcript": "ಪಟ್ಟಿ",
  "uploadFileName": "string",
  "sendEmail": true
}
```

**Response**

```json
{
  "utt_id": "filename",
  "sentence_gop": -0.234,
  "perphone_gop": [["p", -0.1], ["a", 0.2], ["t", -0.3]],
  "latency_ms": 2500
}
```

**Batch Sentence Processing**

- URL: `/api/v1/process-sentence`
- Method: `POST`
- Description: Process multiple audio files for one sentence - runs GOP on all files, finds the best score, then runs OHM on the best file. Returns comprehensive results in a single request.

**Request Body**

```json
{
  "userId": "string",
  "name": "string",
  "communityWorkerName": "string",
  "sentenceId": 1,
  "transcript": "ಪಟ್ಟಿ",
  "language": "kn",
  "uploadFileNames": ["file1.m4a", "file2.m4a", "file3.m4a"],
  "sendEmail": true
}
```

**Response**

```json
{
  "success": true,
  "data": {
    "sentenceId": 1,
    "transcript": "ಪಟ್ಟಿ",
    "totalFiles": 3,
    "gopResults": [
      {
        "filename": "file1.m4a",
        "utt_id": "file1",
        "sentence_gop": -0.234,
        "perphone_gop": [["p", -0.1], ["a", 0.2]],
        "latency_ms": 2500
      }
    ],
    "bestFile": {
      "filename": "file2.m4a",
      "gopScore": -0.123
    },
    "ohmRating": 2.45,
    "errors": null
  },
  "metadata": {
    "requestId": "uuid",
    "processingTime": 12.456,
    "timestamp": 1738176080542,
    "error": null
  }
}
```

## Project Structure

The API is now organized into a modular structure:

```
├── app.py              # Main FastAPI application
├── endpoints/          # Route handlers
│   └── batch.py       # Batch processing endpoints
├── models/            # Pydantic schemas
│   └── schemas.py     # Request/response models
├── services/          # Business logic
│   └── processing.py  # Audio processing services
└── utils/             # Utilities
    └── helpers.py     # Helper functions and validation
```

### Docker commands

- Build Docker Container

```shell
docker build -t cleftcare-ohm .
```

- Run Docker Container

```shell
docker run -p 8080:8080 --name cleftcare-ohm-container cleftcare-ohm
```

Detached (Runs in Background)

```shell
docker run -p 8080:8080 --name cleftcare-ohm-container -d cleftcare-ohm
```

### Google Cloud Run Deployment

- Build for linux/amd64, Tag, and Push

```shell
docker buildx build --platform linux/amd64 -t us-east1-docker.pkg.dev/cleftcare/cleftcare-ohm/cleftcare-ohm:latest --push .
```

- Build Docker Container for AMD64

```shell
docker buildx build --platform linux/amd64 -t cleftcare-ohm .
```

- Tag Container

```shell
docker tag cleftcare-ohm us-east1-docker.pkg.dev/cleftcare/cleftcare-ohm/cleftcare-ohm:latest
```

- Push Tagged Container to Artifact Registry

```shell
docker push us-east1-docker.pkg.dev/cleftcare/cleftcare-ohm/cleftcare-ohm:latest
```

## GOP System Requirements

The `/gop` endpoint requires Kaldi tooling and Kannada models to function properly:

- **Kaldi binaries**: `steps/`, `utils/`, `local/` directories with speech processing scripts
- **Configuration**: `conf/mfcc_hires.conf` for MFCC feature extraction
- **Models**:
  - `models/vosk_kannada_model/` - Pre-trained Kannada acoustic model
  - `models/LM_2gram_aiish/` - 2-gram language model for Kannada
- **Utilities**: `make_text_phone.pl` for phonetic text conversion

## Environment Variables

| Variable                | Description                 | Required |
| ----------------------- | --------------------------- | -------- |
| `AWS_ACCESS_KEY_ID`     | S3 access key               | Yes      |
| `AWS_SECRET_ACCESS_KEY` | S3 secret key               | Yes      |
| `AWS_DEFAULT_REGION`    | S3 region                   | Yes      |
| `MAILJET_API_KEY`       | Email service key           | Optional |
| `MAILJET_API_SECRET`    | Email service secret        | Optional |
| `EMAIL_FROM`            | Sender email                | Optional |
| `EMAIL_FROM_NAME`       | Sender name                 | Optional |
| `API_KEY`               | Authentication API key      | Yes      |
| `PORT`                  | Server port (default: 8080) | No       |

## Deployment Notes

- Current OHM-only image (this README): works with the provided `Dockerfile` and Cloud Run.
- GOP requires Kaldi binaries and support scripts (`steps/`, `utils/`, `local/`, and `conf/mfcc_hires.conf`) plus Kannada models in `models/`.
- Recommended approach for cost and cold starts:
  - Keep OHM as-is on Cloud Run (small image, quick cold start).
  - Deploy GOP as a separate Cloud Run service based on a Kaldi image (e.g., `kaldiasr/kaldi`) and copy required scripts/configs into the image. Point GOP code to those paths.

Apple Silicon (M‑series) note for Cloud Run: build for `linux/amd64` as shown above using `docker buildx`.

## Models

- OHM models: `models/*.pth`, regressors `*.pkl`, normalization `Mean.npy`, `Std.npy`.
- GOP models: `models/vosk_kannada_model/` (acoustic), `models/LM_2gram_aiish/` (language) plus `make_text_phone.pl` and Kaldi support scripts.

## Health Check

- URL: `/`
- Method: `GET`
- Response:

```json
{ "status": "ok" }
```
