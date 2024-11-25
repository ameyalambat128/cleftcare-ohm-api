from transformers import Wav2Vec2Model
import torch
import torch.nn as nn
import boto3
from dotenv import load_dotenv
import os
from tempfile import NamedTemporaryFile
from pathlib import Path

from ohm import predict_ohm_rating


if __name__ == "__main__":
    # S3 Client Setup
    S3_BUCKET_NAME = 'cleftcare-test'
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION')
    )
    s3_key = "MILE_03_SP_0002_UTT_0003.wav"
    with NamedTemporaryFile(delete=True, suffix=".m4a", dir=os.getcwd()) as temp_file:
        temp_file_path = temp_file.name
        s3.download_file(S3_BUCKET_NAME, s3_key, temp_file_path)
        print("Downloaded file from S3: ", temp_file_path)

        # Check if the file is already in .wav format
        if not temp_file_path.endswith(".wav"):
            # Convert .m4a to .wav
            wav_path = temp_file_path.replace(".m4a", ".wav")
            os.system(f"ffmpeg -i {temp_file_path} {wav_path}")
        else:
            wav_path = temp_file_path

    # Process the downloaded file for prediction
    temp_folder = Path(temp_file_path).parent
    perceptual_rating = predict_ohm_rating(temp_folder)
    print(perceptual_rating)
