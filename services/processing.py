import os
import subprocess
import time
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import List, Dict, Tuple

from ohm import predict_ohm_rating
from gop_module import compute_gop


class AudioProcessor:
    def __init__(self, s3_client, bucket_name: str):
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.development_mode = os.getenv("ENVIRONMENT", "production") == "development"

    def download_and_convert_audio(self, upload_file_name: str) -> str:
        """Download audio from S3 or use local file in development mode"""
        audio_dir = os.path.join(os.getcwd(), "audios")
        os.makedirs(audio_dir, exist_ok=True)

        if self.development_mode:
            # Use local file from audios directory; fall back to audios/samples
            primary_path = os.path.join(audio_dir, upload_file_name)
            samples_dir = os.path.join(audio_dir, "samples")
            samples_path = os.path.join(samples_dir, upload_file_name)

            if os.path.exists(primary_path):
                local_file_path = primary_path
            elif os.path.exists(samples_path):
                local_file_path = samples_path
            else:
                raise FileNotFoundError(
                    f"Audio file not found in 'audios/' or 'audios/samples/': {upload_file_name}"
                )

            # Convert to WAV if needed
            if not local_file_path.endswith(".wav"):
                # Write to /tmp (writable) instead of read-only audios/samples
                import tempfile
                base_name = Path(upload_file_name).stem
                temp_wav = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav", prefix=f"{base_name}_", dir="/tmp"
                )
                wav_path = temp_wav.name
                temp_wav.close()
                
                # Convert to 16kHz mono for Kaldi compatibility
                result = subprocess.run([
                    "ffmpeg", "-y", "-i", local_file_path, 
                    "-ar", "16000", "-ac", "1", wav_path
                ], capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
                return wav_path
            else:
                return local_file_path
        else:
            # Production mode: download from S3
            with NamedTemporaryFile(delete=True, suffix=".m4a", dir=audio_dir) as temp_file:
                temp_file_path = temp_file.name
                self.s3_client.download_file(self.bucket_name, upload_file_name, temp_file_path)

                # Convert to WAV if needed
                if not temp_file_path.endswith(".wav"):
                    wav_path = temp_file_path.replace(".m4a", ".wav")
                    # Convert to 16kHz mono for Kaldi compatibility
                    result = subprocess.run([
                        "ffmpeg", "-y", "-i", temp_file_path, 
                        "-ar", "16000", "-ac", "1", wav_path
                    ], capture_output=True, text=True, timeout=30)
                    if result.returncode != 0:
                        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
                else:
                    wav_path = temp_file_path

                return wav_path

    def process_gop(self, wav_path: str, transcript: str) -> Dict:
        """Process a single audio file with GOP"""
        return compute_gop(wav_path, transcript)

    def process_ohm(self, wav_path: str, language: str) -> float:
        """Process a single audio file with OHM"""
        temp_folder = Path(wav_path).parent
        return predict_ohm_rating(temp_folder, language)

    def process_sentence_batch(self, upload_file_names: List[str], transcript: str, language: str) -> Dict:
        """Process multiple audio files for one sentence - find best GOP then get OHM"""
        results = {
            "gop_results": [],
            "best_gop_file": None,
            "best_gop_score": float('-inf'),
            "best_wav_path": None,
            "ohm_rating": None
        }

        wav_files_to_cleanup = []

        # Process each file with GOP
        for filename in upload_file_names:
            try:
                wav_path = self.download_and_convert_audio(filename)
                wav_files_to_cleanup.append(wav_path)
                
                gop_result = self.process_gop(wav_path, transcript)

                gop_result["filename"] = filename
                results["gop_results"].append(gop_result)

                # Track best GOP score
                gop_score = gop_result.get("sentence_gop", float('-inf'))
                if gop_score > results["best_gop_score"]:
                    results["best_gop_score"] = gop_score
                    results["best_gop_file"] = filename
                    results["best_wav_path"] = wav_path

            except Exception as e:
                # Add error to results but continue processing other files
                results["gop_results"].append({
                    "filename": filename,
                    "error": str(e),
                    "sentence_gop": float('-inf')
                })

        # Process best file with OHM if we found a valid one
        if results["best_wav_path"] and results["best_gop_file"]:
            try:
                # Re-download and convert the best file for OHM processing
                best_wav_path = self.download_and_convert_audio(results["best_gop_file"])
                wav_files_to_cleanup.append(best_wav_path)
                results["ohm_rating"] = self.process_ohm(best_wav_path, language)

            except Exception as e:
                results["ohm_error"] = str(e)

        # Clean up all wav files at the end
        for wav_path in wav_files_to_cleanup:
            try:
                if os.path.exists(wav_path) and wav_path.startswith("/tmp/"):
                    os.remove(wav_path)
            except Exception:
                pass  # Ignore cleanup errors

        return results