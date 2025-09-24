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

**Predict OHM Rating**

- URL: `/predict`
- Method: `POST`
- Description: Accepts a POST request with `userId` and `uploadFileName`, downloads the specified audio file from S3, processes it, and returns the perceptual rating.

**Request Body**

```json
{
  "userId": "string",
  "uploadFileName": "string"
}
```

**Response**

```json
{
  "perceptualRating": <rating>
}
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

## Additional Endpoints (GOP + Combined)

The API can be extended with Kannada GOP (Goodness of Pronunciation) and a Combined assessment (OHM + GOP). These require Kaldi tooling and Kannada models.

### GOP Assessment (Kannada)

- URL: `/gop/predict`
- Method: `POST`
- Description: Upload a WAV file and the expected transcript to compute sentence-level and per-phone GOP.

Form Data:

```
wav_file: <WAV file>
transcript: <string>
```

Example Response:

```json
{
  "service": "gop",
  "utt_id": "input",
  "sentence_gop": -0.234,
  "perphone_gop": [
    ["p", -0.1],
    ["a", 0.2]
  ],
  "latency_ms": 2500,
  "transcript": "ಪಟ್ಟಿ"
}
```

### Combined Assessment (OHM + GOP)

- URL: `/assess`
- Method: `POST`
- Description: Downloads audio from S3 (same inputs as OHM) and returns both OHM and GOP results in one response.

Request Body (JSON):

```json
{
  "userId": "string",
  "name": "string",
  "communityWorkerName": "string",
  "promptNumber": 1,
  "language": "en|kn",
  "uploadFileName": "string",
  "sendEmail": true,
  "transcript": "optional transcript for GOP"
}
```

Example Response:

```json
{
  "userId": "patient123",
  "name": "John Doe",
  "language": "kn",
  "assessments": {
    "ohm": {
      "perceptualRating": 2.45,
      "description": "Oral Hypernasality Measure for cleft lip/palate assessment"
    },
    "gop": {
      "sentence_gop": -0.234,
      "perphone_gop": [["p", -0.1]],
      "transcript": "ಪಟ್ಟಿ",
      "description": "Goodness of Pronunciation for general speech quality"
    }
  },
  "timestamp": 2500
}
```

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
