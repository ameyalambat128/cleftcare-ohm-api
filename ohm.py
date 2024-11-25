import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import joblib  # For loading the regressor model
import __main__
# Paths (Update as per your environment)
FEATURE_SAVE_DIR = './Test_spkr_folder'  # Directory to save features
os.makedirs(FEATURE_SAVE_DIR, exist_ok=True)


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the DNN Model architecture


class DNNModel(nn.Module):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 1024)
        self.relu3 = nn.ReLU()
        # Output layer with 2 classes: nasal and oral
        self.fc4 = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

# Load trained models and normalization parameters


model_path = './models/english_xlsr_librispeech_model_32batch_100h_valid.pth'
regressor_path = './models/regressor_model.pkl'  # Path to saved regressor model
mean_path = './models/Mean.npy'  # Path to normalization mean
std_path = './models/Std.npy'  # Path to normalization std

# Load normalization parameters
feats_m1 = np.load(mean_path)
feats_s1 = np.load(std_path)


# Load regressor
regressor = joblib.load(regressor_path)


def load_dnn_model(model_path, device):
    try:
        __main__.DNNModel = DNNModel
        print("Attempting to load the pre-trained DNN model...")
        dnn_model = torch.load(model_path, map_location=device)
        dnn_model.eval()
        print("DNN model loaded successfully!")
        return dnn_model
    except Exception as e:
        print(f"Error loading DNN model: {e}")
        raise

# Feature extraction model


class Wav2Vec2FeatureExtractorModel(nn.Module):
    def __init__(self, base_model):
        super(Wav2Vec2FeatureExtractorModel, self).__init__()
        self.base_model = base_model

    def forward(self, input_values):
        outputs = self.base_model(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states[11]
        print(f"Shape after extracting hidden states: {hidden_states.shape}")
        hidden_states = hidden_states.squeeze(0)
        print(f"Shape after squeezing batch dimension: {hidden_states.shape}")
        feature_vector = hidden_states.transpose(0, 1)
        print(f"Shape after transposing: {feature_vector.shape}")
        return feature_vector


wav2vec_model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53")
feature_extractor_model = Wav2Vec2FeatureExtractorModel(
    wav2vec_model).to(device)
feature_extractor_model.eval()

# Function to extract features for a single audio file


def extract_features(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000)
    print(f"Shape of loaded audio: {audio.shape}")
    inputs = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53")(audio, return_tensors="pt", sampling_rate=16000)
    input_values = inputs.input_values.to(device)
    print(f"Shape of input_values: {input_values.shape}")
    with torch.no_grad():
        features = feature_extractor_model(input_values).cpu().numpy()
    print(f"Shape of extracted features: {features.shape}")
    return features

# Function to compute OHM for a single feature file


def compute_ohm(features, feats_m1, feats_s1):
    print(f"Initial feature shape: {features.shape}")
    features = features.T  # Transpose features
    feats_m1 = feats_m1.squeeze()
    feats_s1 = feats_s1.squeeze()
    print(f"Shape after transpose: {features.shape}")
    features = (features - feats_m1) / feats_s1  # Normalize
    print(f"Shape after normalization: {features.shape}")
    features = torch.tensor(features, dtype=torch.float32).to(
        device)  # Shape: (sequence_length, 1024)
    print(f"Shape after converting to tensor: {features.shape}")
    with torch.no_grad():
        dnn_model = load_dnn_model(model_path, device)
        outputs = dnn_model(features)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
    print(f"Shape of probabilities: {probabilities.shape}")
    p_n = probabilities[:, 0]  # Probability for nasal
    ohm = np.cbrt(p_n)  # Cube root of probabilities for nasal class
    print(f"Computed OHM values: {ohm}")
    return np.mean(ohm)

# Function to process a folder of audio files for a single speaker


def process_speaker_folder(speaker_folder):
    ohm_scores = []
    for audio_file in tqdm(os.listdir(speaker_folder), desc=f"Processing speaker folder: {speaker_folder}"):
        audio_path = os.path.join(speaker_folder, audio_file)
        if not audio_file.endswith('.wav') or not os.path.isfile(audio_path):
            continue
        features = extract_features(audio_path)  # Extract features
        ohm = compute_ohm(features, feats_m1, feats_s1)  # Compute OHM
        ohm_scores.append(ohm)
    return np.mean(ohm_scores) if ohm_scores else None

# Function to predict perceptual OHM rating for a speaker


def predict_ohm_rating(speaker_folder):
    avg_ohm = process_speaker_folder(speaker_folder)
    if avg_ohm is not None:
        avg_ohm = np.array([[avg_ohm]])
        perceptual_rating = regressor.predict(avg_ohm)[0]
        return perceptual_rating
    return None


# Example usage
if __name__ == "__main__":
    # Update to the folder path of a given speaker
    speaker_folder_path = './Test_spkr_folder'
    perceptual_rating = predict_ohm_rating(speaker_folder_path)
    if perceptual_rating is not None:
        print(
            f"Predicted Perceptual OHM Rating for speaker: {perceptual_rating:.2f}")
    else:
        print("No valid OHM scores computed for the speaker.")
