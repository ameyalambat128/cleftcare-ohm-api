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
docker run -p 8000:8000 --name cleftcare-ohm-container cleftcare-ohm
```

Detached (Runs in Background)

```shell
docker run -p 8000:8000 --name cleftcare-ohm-container -d cleftcare-ohm
```

### Google Cloud Run Deployment

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
