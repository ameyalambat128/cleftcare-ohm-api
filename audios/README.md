# Audio Files for Development Testing

## Directory Structure

```
audios/
├── README.md          # This file
├── samples/           # Sample audio files for testing
└── [your-files]       # Place your test audio files here
```

## Usage in Development Mode

1. **Place your audio files** directly in the `audios/` directory
2. **Supported formats**: `.wav`, `.m4a`, `.mp3`, `.flac`
3. **File naming**: Use descriptive names like `kannada_sentence_1.m4a`

## Sample Files

The `samples/` directory contains example audio files for immediate testing. You can:
- Use these for initial API testing
- Reference them as examples for your own recordings
- Copy them to the main `audios/` directory for testing

## Testing Examples

### Individual Endpoints

```bash
# Test OHM endpoint
curl -X POST http://localhost:8080/ohm \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-12345" \
  -d '{
    "userId": "test-user",
    "name": "Test User",
    "communityWorkerName": "Test Worker",
    "promptNumber": 1,
    "language": "kn",
    "uploadFileName": "sample_kannada.m4a",
    "sendEmail": false
  }'

# Test GOP endpoint
curl -X POST http://localhost:8080/gop \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-12345" \
  -d '{
    "userId": "test-user",
    "name": "Test User",
    "communityWorkerName": "Test Worker",
    "transcript": "ಪಟ್ಟಿ",
    "uploadFileName": "sample_kannada.m4a",
    "sendEmail": false
  }'
```

### Batch Processing

```bash
# Test batch sentence processing
curl -X POST http://localhost:8080/api/v1/process-sentence \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-12345" \
  -d '{
    "userId": "test-user",
    "name": "Test User",
    "communityWorkerName": "Test Worker",
    "sentenceId": 1,
    "transcript": "ಪಟ್ಟಿ",
    "language": "kn",
    "uploadFileNames": ["sample1.m4a", "sample2.m4a", "sample3.m4a"],
    "sendEmail": false
  }'
```

## Notes

- **Development Mode**: When `ENVIRONMENT=development`, the API will look for files in this directory instead of downloading from S3
- **File Conversion**: The API automatically converts audio files to WAV format as needed
- **Cleanup**: Temporary WAV files are automatically cleaned up after processing