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
docker run -p 8000:8000 cleftcare-ohm
```
