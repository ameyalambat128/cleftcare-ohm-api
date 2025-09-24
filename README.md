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
