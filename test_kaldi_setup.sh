#!/bin/bash
# Test script to validate Kaldi setup in the container

set -e

echo "=== Testing Kaldi Setup ==="

# Source the environment
source ./path.sh

echo "1. Testing Kaldi binaries..."
echo "Checking for compute-gop binary:"
which compute-gop || echo "compute-gop not found in PATH"

echo "Checking for other required binaries:"
which ali-to-phones || echo "ali-to-phones not found"
which compile-train-graphs-without-lexicon || echo "compile-train-graphs-without-lexicon not found"

echo "2. Testing required directories..."
[ -d "steps" ] && echo "✓ steps/ directory found" || echo "✗ steps/ directory missing"
[ -d "utils" ] && echo "✓ utils/ directory found" || echo "✗ utils/ directory missing"
[ -d "local" ] && echo "✓ local/ directory found" || echo "✗ local/ directory missing"
[ -d "conf" ] && echo "✓ conf/ directory found" || echo "✗ conf/ directory missing"

echo "3. Testing configuration files..."
[ -f "conf/mfcc_hires.conf" ] && echo "✓ mfcc_hires.conf found" || echo "✗ mfcc_hires.conf missing"
[ -f "make_text_phone.pl" ] && echo "✓ make_text_phone.pl found" || echo "✗ make_text_phone.pl missing"
[ -f "local/remove_phone_markers.pl" ] && echo "✓ remove_phone_markers.pl found" || echo "✗ remove_phone_markers.pl missing"

echo "4. Testing model directories..."
[ -d "models/vosk_kannada_model" ] && echo "✓ Kannada acoustic model found" || echo "✗ Kannada acoustic model missing"
[ -d "models/LM_2gram_aiish" ] && echo "✓ Language model found" || echo "✗ Language model missing"

echo "5. Testing required scripts..."
[ -f "utils/utt2spk_to_spk2utt.pl" ] && echo "✓ utt2spk_to_spk2utt.pl found" || echo "✗ utt2spk_to_spk2utt.pl missing"
[ -f "steps/make_mfcc.sh" ] && echo "✓ make_mfcc.sh found" || echo "✗ make_mfcc.sh missing"
[ -f "steps/compute_cmvn_stats.sh" ] && echo "✓ compute_cmvn_stats.sh found" || echo "✗ compute_cmvn_stats.sh missing"

echo "=== Kaldi Setup Test Complete ==="

echo "Environment variables:"
echo "KALDI_ROOT: $KALDI_ROOT"
echo "PATH: $PATH"