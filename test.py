from ohm import (
    load_dnn_model,
    load_normalization_params,
    load_regressor,
    Wav2Vec2FeatureExtractorModel,
    predict_ohm_rating,
)
from transformers import Wav2Vec2Model
import torch
import torch.nn as nn


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


def test_predict_ohm_rating():
    """Tests the predict_ohm_rating function from ohm.py."""
    # Define paths
    speaker_folder_path = './Test_spkr_folder'
    model_path = './models/english_xlsr_librispeech_model_32batch_100h_valid.pth'
    regressor_path = './models/regressor_model.pkl'
    mean_path = './models/Mean.npy'
    std_path = './models/Std.npy'

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models and parameters
    print("Loading DNN model...")
    dnn_model = load_dnn_model(model_path, device)

    print("Loading normalization parameters...")
    feats_m1, feats_s1 = load_normalization_params(mean_path, std_path)

    print("Loading regressor model...")
    regressor = load_regressor(regressor_path)

    print("Loading Wav2Vec2 feature extractor model...")
    wav2vec_model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53")
    feature_extractor_model = Wav2Vec2FeatureExtractorModel(
        wav2vec_model).to(device)
    feature_extractor_model.eval()

    # Run prediction
    print("Predicting perceptual OHM rating...")
    perceptual_rating = predict_ohm_rating(
        speaker_folder_path, feature_extractor_model, dnn_model, feats_m1, feats_s1, regressor, device
    )

    # Check the output
    if perceptual_rating is not None:
        print(
            f"Test Passed! Predicted Perceptual OHM Rating: {perceptual_rating:.2f}")
    else:
        print("Test Failed: No valid OHM scores computed for the speaker.")

    return perceptual_rating


if __name__ == "__main__":
    rating = test_predict_ohm_rating()
    print(rating)
