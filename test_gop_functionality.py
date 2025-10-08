#!/usr/bin/env python3
"""
Test script to validate GOP functionality in the container.
This can be run inside the Docker container to test the GOP system.
"""

import os
import sys
import tempfile
import shutil
from gop_module import compute_gop

def create_test_audio():
    """Create a dummy WAV file for testing (placeholder)"""
    # In a real test, you'd want to use a actual audio file
    # For now, we'll just test if the function handles missing files gracefully
    return "/tmp/test_audio.wav"

def test_gop_basic():
    """Basic test of GOP functionality"""
    print("=== Testing GOP Module ===")

    # Test 1: Check if required models exist
    print("\n1. Checking model directories...")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_dirs = [
        "models/vosk_kannada_model/am",
        "models/vosk_kannada_model/graph",
        "models/vosk_kannada_model/ivector",
        "models/LM_2gram_aiish"
    ]

    for model_dir in model_dirs:
        full_path = os.path.join(base_dir, model_dir)
        if os.path.exists(full_path):
            print(f"✓ {model_dir} found")
        else:
            print(f"✗ {model_dir} missing")

    # Test 2: Check if configuration files exist
    print("\n2. Checking configuration files...")
    config_files = [
        "conf/mfcc_hires.conf",
        "make_text_phone.pl",
        "local/remove_phone_markers.pl",
        "path.sh",
        "cmd.sh"
    ]

    for config_file in config_files:
        full_path = os.path.join(base_dir, config_file)
        if os.path.exists(full_path):
            print(f"✓ {config_file} found")
        else:
            print(f"✗ {config_file} missing")

    # Test 3: Try to run GOP with a dummy input (will likely fail gracefully)
    print("\n3. Testing GOP function call...")
    try:
        # Use a simple Kannada transcript for testing
        test_transcript = "ಅಮ್ಮ"  # Simple Kannada word
        dummy_wav = "/tmp/nonexistent.wav"

        print(f"Calling compute_gop with transcript: {test_transcript}")
        result = compute_gop(dummy_wav, test_transcript)

        if "error" in result:
            print(f"✓ GOP function handled error gracefully: {result['error']}")
        else:
            print(f"✓ GOP function completed successfully: {result}")

    except Exception as e:
        print(f"✗ GOP function failed with exception: {str(e)}")

    print("\n=== GOP Test Complete ===")

if __name__ == "__main__":
    test_gop_basic()