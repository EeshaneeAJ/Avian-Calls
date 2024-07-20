from flask import Flask, request, render_template, redirect
import os
import numpy as np
import librosa
import pickle
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
from shapely.geometry import Point
import joblib

app = Flask(__name__)

# Load the model, label encoder, and scaler
model_path = "C:\\Users\\Admin\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\bird_sound_model_hybrid.pkl"
label_encoder_path = "C:\\Users\\Admin\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\label_encoder.pkl"
scaler_path = "C:\\Users\\Admin\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\scaler.pkl"
excel_path = "C:\\Users\\Admin\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\Birds_info.xlsx"

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(label_encoder_path, 'rb') as file:
    label_encoder = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# Load bird details from Excel
bird_details_path = os.path.join(app.root_path, 'Birds_info.xlsx')
bird_details = pd.read_excel(bird_details_path)

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

def highlight_countries(bird_info):
    countries = bird_info['Origin'].split(',')
    countries = [country.strip() for country in countries]

    world = gpd.read_file('data/ne_110m_admin_0_countries.shp')

    bird_map = folium.Map(location=[20, 0], zoom_start=2)

    for country in countries:
        country_shape = world[world['NAME'] == country]
        if not country_shape.empty:
            geo_j = folium.GeoJson(country_shape, style_function=lambda x: {'fillColor': 'orange'})
            geo_j.add_to(bird_map)

    return bird_map

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/explore')
def explore():
    birds = bird_details.to_dict(orient='records')
    return render_template('explore.html', birds=birds)

@app.route('/bird/<bird_name>')
def bird_details_route(bird_name):
    bird_info = bird_details[bird_details['Bird Name'].str.strip() == bird_name.strip()]
    if bird_info.empty:
        return render_template('explore.html', error_message=f'No information found for {bird_name}')

    bird_info = bird_info.iloc[0].to_dict()

    # Correctly format the audio file path
    bird_info['Audio'] = bird_info['Audio'].replace('\\', '/')

    # Generate the map with highlighted countries
    bird_map = highlight_countries(bird_info)

    # Save the map in the static directory
    map_path = os.path.join(app.root_path, 'static', 'map.html')
    bird_map.save(map_path)

    print(bird_info['Audio'])  # Debugging

    return render_template('result.html', bird_info=bird_info)

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded file to a temporary location
        file_path = os.path.join('temp', file.filename)
        file.save(file_path)

        # Load the SVM model and other necessary components
        svm_model = joblib.load(model_path)
        label_encoder = joblib.load('label_encoder.pkl')
        scaler = joblib.load('scaler.pkl')

        # Extract features from the audio file (you need to define extract_features)
        features = features_extractor(file_path)

        # Scale the features
        features_scaled = scaler.transform([features])

        # Predict the bird species
        prediction = svm_model.predict(features_scaled)
        predicted_bird = label_encoder.inverse_transform(prediction)[0]

        # Fetch bird details from the Excel sheet
        bird_info = bird_details[bird_details['Bird Name'].str.strip() == predicted_bird.strip()]
        if bird_info.empty:
            return render_template('result.html', error_message=f'No information found for {predicted_bird}')

        bird_info = bird_info.iloc[0].to_dict()

        # Correctly format the audio file path
        bird_info['Audio'] = bird_info['Audio'].replace('\\', '/')

        # Generate the map with highlighted countries
        bird_map = highlight_countries(bird_info)

        # Save the map in the static directory
        map_path = os.path.join(app.root_path, 'static', 'map.html')
        bird_map.save(map_path)

        return render_template('result.html', bird_info=bird_info)

    return render_template('result.html', error_message='Failed to process the audio file.')

if __name__ == "__main__":
    app.run(debug=True)
