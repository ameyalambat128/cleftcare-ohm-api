import os
import torch
import torch.nn as nn
import numpy as np
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import joblib  # For loading the regressor model


# Paths (Update as per your environment)
FEATURE_SAVE_DIR = './Test_spkr_folder'
MODEL_PATH = './models/english_xlsr_librispeech_model_32batch_100h_valid.pth'
REGRESSOR_PATH = './models/regressor_model.pkl'
MEAN_PATH = './models/Mean.npy'
STD_PATH = './models/Std.npy'

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.fc4 = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x


class Wav2Vec2FeatureExtractorModel(nn.Module):
    """Feature extraction model based on Wav2Vec2."""

    def __init__(self, base_model):
        super(Wav2Vec2FeatureExtractorModel, self).__init__()
        self.base_model = base_model

    def forward(self, input_values):
        outputs = self.base_model(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states[11]
        return hidden_states.squeeze(0).transpose(0, 1)


def load_dnn_model(model_path, device):
    """Loads the pre-trained DNN model."""
    try:
        print("Loading the pre-trained DNN model...")
        dnn_model = torch.load(model_path, map_location=device)
        dnn_model.eval()
        print("DNN model loaded successfully!")
        return dnn_model
    except Exception as e:
        raise RuntimeError(f"Error loading DNN model: {e}")


def load_normalization_params(mean_path, std_path):
    """Loads normalization parameters (mean and std)."""
    try:
        print("Loading normalization parameters...")
        feats_m1 = np.load(mean_path)
        feats_s1 = np.load(std_path)
        print("Normalization parameters loaded successfully!")
        return feats_m1, feats_s1
    except Exception as e:
        raise RuntimeError(f"Error loading normalization parameters: {e}")


def load_regressor(regressor_path):
    """Loads the trained regressor model."""
    try:
        print("Loading the regressor model...")
        regressor = joblib.load(regressor_path)
        print("Regressor model loaded successfully!")
        return regressor
    except Exception as e:
        raise RuntimeError(f"Error loading regressor model: {e}")


def extract_features(audio_path, feature_extractor_model, device):
    """Extracts features for a single audio file."""
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53"
    )(audio, return_tensors="pt", sampling_rate=16000)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        features = feature_extractor_model(input_values).cpu().numpy()
    return features


def compute_ohm(features, feats_m1, feats_s1, dnn_model, device):
    """Computes OHM for a single feature file."""
    features = features.T  # Transpose features
    features = (features - feats_m1.squeeze()) / \
        feats_s1.squeeze()  # Normalize
    features = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = dnn_model(features)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
    p_n = probabilities[:, 0]  # Probability for nasal
    ohm = np.cbrt(p_n)  # Cube root of probabilities for nasal class
    return np.mean(ohm)


def process_speaker_folder(speaker_folder, feature_extractor_model, dnn_model, feats_m1, feats_s1, device):
    """Processes a folder of audio files for a single speaker."""
    ohm_scores = []
    for audio_file in tqdm(os.listdir(speaker_folder), desc="Processing speaker folder"):
        audio_path = os.path.join(speaker_folder, audio_file)
        if not audio_file.endswith('.wav') or not os.path.isfile(audio_path):
            continue
        features = extract_features(
            audio_path, feature_extractor_model, device)
        ohm = compute_ohm(features, feats_m1, feats_s1, dnn_model, device)
        ohm_scores.append(ohm)
    return np.mean(ohm_scores) if ohm_scores else None


def predict_ohm_rating(speaker_folder, feature_extractor_model, dnn_model, feats_m1, feats_s1, regressor, device):
    """Predicts the perceptual OHM rating for a speaker."""
    avg_ohm = process_speaker_folder(
        speaker_folder, feature_extractor_model, dnn_model, feats_m1, feats_s1, device)
    if avg_ohm is not None:
        avg_ohm = np.array([[avg_ohm]])
        perceptual_rating = regressor.predict(avg_ohm)[0]
        return perceptual_rating
    return None


# Main execution
if __name__ == "__main__":
    # Load models and parameters
    dnn_model = load_dnn_model(MODEL_PATH, DEVICE)
    feats_m1, feats_s1 = load_normalization_params(MEAN_PATH, STD_PATH)
    regressor = load_regressor(REGRESSOR_PATH)

    # Load Wav2Vec2 feature extractor
    wav2vec_model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53")
    feature_extractor_model = Wav2Vec2FeatureExtractorModel(
        wav2vec_model).to(DEVICE)
    feature_extractor_model.eval()

    # Update to the folder path of a given speaker
    speaker_folder_path = FEATURE_SAVE_DIR
    perceptual_rating = predict_ohm_rating(
        speaker_folder_path, feature_extractor_model, dnn_model, feats_m1, feats_s1, regressor, DEVICE
    )

    if perceptual_rating is not None:
        print(
            f"Predicted Perceptual OHM Rating for speaker: {perceptual_rating:.2f}")
    else:
        print("No valid OHM scores computed for the speaker.")
