import os
import numpy as np
import librosa
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the model, label encoder, and scaler
model_path = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\bird_sound_model_hybrid.pkl"
label_encoder_path = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\label_encoder.pkl"
scaler_path = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\scaler.pkl"

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(label_encoder_path, 'rb') as file:
    label_encoder = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# Function to extract features from audio files
def features_extractor(file):
    try:
        audio, sample_rate = librosa.load(file, sr=None, mono=True, res_type='kaiser_fast')
        if audio.size == 0:
            print(f"Empty audio file: {file}")
            return np.array([])  # Return empty array if audio is empty

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
        mfccs_scaled_features = np.mean(mfccs.T, axis=0)

        # Extract Chroma feature
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_scaled_features = np.mean(chroma.T, axis=0)

        # Extract Spectral Contrast feature
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        spectral_contrast_scaled_features = np.mean(spectral_contrast.T, axis=0)

        # Extract Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        mel_spectrogram_scaled_features = np.mean(mel_spectrogram.T, axis=0)

        # Extract Zero Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
        zero_crossing_rate_scaled_features = np.mean(zero_crossing_rate.T, axis=0)

        # Extract Root Mean Square Energy
        rmse = librosa.feature.rms(y=audio)
        rmse_scaled_features = np.mean(rmse.T, axis=0)

        # Combine features
        combined_features = np.hstack((mfccs_scaled_features, chroma_scaled_features, spectral_contrast_scaled_features, mel_spectrogram_scaled_features, zero_crossing_rate_scaled_features, rmse_scaled_features))
        return combined_features
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return np.array([])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})

    if file:
        # Save the file to a temporary location
        file_path = os.path.join("C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\temp", file.filename)
        file.save(file_path)

        # Extract features
        features = features_extractor(file_path)
        if features.shape == (0,):
            return jsonify({"error": "Could not extract features from the file"})

        # Scale features
        features_scaled = scaler.transform([features])

        # Predict
        prediction = model.predict(features_scaled)
        species = label_encoder.inverse_transform(prediction)[0]  # Assuming label_encoder is part of the model

        return render_template('result.html', species=species)

if __name__ == '__main__':
    app.run(debug=True)
