# import os
# import numpy as np
# import librosa
# import pickle
# from flask import Flask, request, jsonify, render_template


# app = Flask(__name__)

# # Load the model, label encoder, and scaler
# model_path = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\bird_sound_model_hybrid.pkl"
# label_encoder_path = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\label_encoder.pkl"
# scaler_path = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\scaler.pkl"

# with open(model_path, 'rb') as file:
#     model = pickle.load(file)

# with open(label_encoder_path, 'rb') as file:
#     label_encoder = pickle.load(file)

# with open(scaler_path, 'rb') as file:
#     scaler = pickle.load(file)

# # Function to extract features from audio files
# def features_extractor(file):
#     try:
#         audio, sample_rate = librosa.load(file, sr=None, mono=True, res_type='kaiser_fast')
#         if audio.size == 0:
#             print(f"Empty audio file: {file}")
#             return np.array([])  # Return empty array if audio is empty

#         # Extract MFCCs
#         mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
#         mfccs_scaled_features = np.mean(mfccs.T, axis=0)

#         # Extract Chroma feature
#         chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
#         chroma_scaled_features = np.mean(chroma.T, axis=0)

#         # Extract Spectral Contrast feature
#         spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
#         spectral_contrast_scaled_features = np.mean(spectral_contrast.T, axis=0)

#         # Extract Mel Spectrogram
#         mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
#         mel_spectrogram_scaled_features = np.mean(mel_spectrogram.T, axis=0)

#         # Extract Zero Crossing Rate
#         zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
#         zero_crossing_rate_scaled_features = np.mean(zero_crossing_rate.T, axis=0)

#         # Extract Root Mean Square Energy
#         rmse = librosa.feature.rms(y=audio)
#         rmse_scaled_features = np.mean(rmse.T, axis=0)

#         # Combine features
#         combined_features = np.hstack((mfccs_scaled_features, chroma_scaled_features, spectral_contrast_scaled_features, mel_spectrogram_scaled_features, zero_crossing_rate_scaled_features, rmse_scaled_features))
#         return combined_features
#     except Exception as e:
#         print(f"Error processing {file}: {e}")
#         return np.array([])

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file provided"})

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"})

#     if file:
#         # Save the file to a temporary location
#         file_path = os.path.join("C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\temp", file.filename)
#         file.save(file_path)

#         # Extract features
#         features = features_extractor(file_path)
#         if features.shape == (0,):
#             return jsonify({"error": "Could not extract features from the file"})

#         # Scale features
#         features_scaled = scaler.transform([features])

#         # Predict
#         prediction = model.predict(features_scaled)
#         species = label_encoder.inverse_transform(prediction)[0]  # Assuming label_encoder is part of the model

#         return render_template('result.html', species=species)

# if __name__ == '__main__':
#     app.run(debug=True)

# import os
# import numpy as np
# import librosa
# import pickle
# from flask import Flask, request, jsonify, render_template
# import wikipedia

# app = Flask(__name__)

# # Load the model, label encoder, and scaler
# model_path = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\bird_sound_model_hybrid.pkl"
# label_encoder_path = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\label_encoder.pkl"
# scaler_path = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\scaler.pkl"

# with open(model_path, 'rb') as file:
#     model = pickle.load(file)

# with open(label_encoder_path, 'rb') as file:
#     label_encoder = pickle.load(file)

# with open(scaler_path, 'rb') as file:
#     scaler = pickle.load(file)

# def features_extractor(file):
#     try:
#         audio, sample_rate = librosa.load(file, sr=None, mono=True, res_type='kaiser_fast')
#         if audio.size == 0:
#             print(f"Empty audio file: {file}")
#             return np.array([])  # Return empty array if audio is empty

#         # Extract MFCCs
#         mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
#         mfccs_scaled_features = np.mean(mfccs.T, axis=0)

#         # Extract Chroma feature
#         chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
#         chroma_scaled_features = np.mean(chroma.T, axis=0)

#         # Extract Spectral Contrast feature
#         spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
#         spectral_contrast_scaled_features = np.mean(spectral_contrast.T, axis=0)

#         # Extract Mel Spectrogram
#         mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
#         mel_spectrogram_scaled_features = np.mean(mel_spectrogram.T, axis=0)

#         # Extract Zero Crossing Rate
#         zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
#         zero_crossing_rate_scaled_features = np.mean(zero_crossing_rate.T, axis=0)

#         # Extract Root Mean Square Energy
#         rmse = librosa.feature.rms(y=audio)
#         rmse_scaled_features = np.mean(rmse.T, axis=0)

#         # Combine features
#         combined_features = np.hstack((mfccs_scaled_features, chroma_scaled_features, spectral_contrast_scaled_features, mel_spectrogram_scaled_features, zero_crossing_rate_scaled_features, rmse_scaled_features))
#         return combined_features
#     except Exception as e:
#         print(f"Error processing {file}: {e}")
#         return np.array([])

# def get_bird_information(bird_name):
#     try:
#         page = wikipedia.page(bird_name)
#         return {
#             'scientific_name': page.title,
#             'summary': page.summary,
#             'url': page.url
#         }
#     except wikipedia.exceptions.DisambiguationError as e:
#         suggestions = e.options[:5]
#         return {
#             'error': f"Ambiguous bird name. Suggestions: {', '.join(suggestions)}"
#         }
#     except wikipedia.exceptions.PageError:
#         return {
#             'error': "Bird information not found on Wikipedia."
#         }

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return render_template('index.html', error_message="No file provided")

#         file = request.files['file']
#         if file.filename == '':
#             return render_template('index.html', error_message="No file selected")

#         if file:
#             file_path = os.path.join("C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\temp", file.filename)
#             file.save(file_path)

#             features = features_extractor(file_path)
#             os.remove(file_path)

#             if features.shape == (0,):
#                 return render_template('index.html', error_message="Could not extract features from the file")

#             features_scaled = scaler.transform([features])
#             prediction = model.predict(features_scaled)
#             species = label_encoder.inverse_transform(prediction)[0]

#             bird_info = get_bird_information(species)
#             if 'error' in bird_info:
#                 return render_template('result.html', error_message=bird_info['error'])
#             else:
#                 return render_template('result.html', bird_info=bird_info)

#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)
# import os
# import numpy as np
# import librosa
# import pickle
# from flask import Flask, request, jsonify, render_template
# import wikipedia

# app = Flask(__name__)

# # Load the model, label encoder, and scaler
# model_path = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\bird_sound_model_hybrid.pkl"
# label_encoder_path = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\label_encoder.pkl"
# scaler_path = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\scaler.pkl"

# with open(model_path, 'rb') as file:
#     model = pickle.load(file)

# with open(label_encoder_path, 'rb') as file:
#     label_encoder = pickle.load(file)

# with open(scaler_path, 'rb') as file:
#     scaler = pickle.load(file)

# def features_extractor(file):
#     try:
#         audio, sample_rate = librosa.load(file, sr=None, mono=True, res_type='kaiser_fast')
#         if audio.size == 0:
#             print(f"Empty audio file: {file}")
#             return np.array([])  # Return empty array if audio is empty

#         # Extract MFCCs
#         mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
#         mfccs_scaled_features = np.mean(mfccs.T, axis=0)

#         # Extract Chroma feature
#         chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
#         chroma_scaled_features = np.mean(chroma.T, axis=0)

#         # Extract Spectral Contrast feature
#         spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
#         spectral_contrast_scaled_features = np.mean(spectral_contrast.T, axis=0)

#         # Extract Mel Spectrogram
#         mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
#         mel_spectrogram_scaled_features = np.mean(mel_spectrogram.T, axis=0)

#         # Extract Zero Crossing Rate
#         zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
#         zero_crossing_rate_scaled_features = np.mean(zero_crossing_rate.T, axis=0)

#         # Extract Root Mean Square Energy
#         rmse = librosa.feature.rms(y=audio)
#         rmse_scaled_features = np.mean(rmse.T, axis=0)

#         # Combine features
#         combined_features = np.hstack((mfccs_scaled_features, chroma_scaled_features, spectral_contrast_scaled_features, mel_spectrogram_scaled_features, zero_crossing_rate_scaled_features, rmse_scaled_features))
#         return combined_features
#     except Exception as e:
#         print(f"Error processing {file}: {e}")
#         return np.array([])

# def get_bird_information(bird_name):
#     try:
#         summary = wikipedia.summary(bird_name, sentences=2)
#         page = wikipedia.page(bird_name)
#         images = page.images

#         # Get the first valid image URL
#         image_url = next((img for img in images if img.endswith(('jpg', 'jpeg', 'png', 'gif'))), None)
        
#         return {
#             'scientific_name': page.title,
#             'summary': summary,
#             'image_url': image_url,
#             'url': page.url
#         }
#     except wikipedia.exceptions.DisambiguationError as e:
#         suggestions = e.options[:5]
#         return {
#             'error': f"Ambiguous bird name. Suggestions: {', '.join(suggestions)}"
#         }
#     except wikipedia.exceptions.PageError:
#         return {
#             'error': "Bird information not found on Wikipedia."
#         }

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return render_template('index.html', error_message="No file provided")

#         file = request.files['file']
#         if file.filename == '':
#             return render_template('index.html', error_message="No file selected")

#         if file:
#             file_path = os.path.join("C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\temp", file.filename)
#             file.save(file_path)

#             features = features_extractor(file_path)
#             os.remove(file_path)

#             if features.shape == (0,):
#                 return render_template('index.html', error_message="Could not extract features from the file")

#             features_scaled = scaler.transform([features])
#             prediction = model.predict(features_scaled)
#             species = label_encoder.inverse_transform(prediction)[0]

#             bird_info = get_bird_information(species)
#             if 'error' in bird_info:
#                 return render_template('result.html', error_message=bird_info['error'])
#             else:
#                 return render_template('result.html', bird_info=bird_info)

#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)
'''import os
import numpy as np
import librosa
import pickle
import pandas as pd
from flask import Flask, request, render_template
import folium

app = Flask(__name__)

# Load the model, label encoder, and scaler
model_path = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\bird_sound_model_hybrid.pkl"
label_encoder_path = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\label_encoder.pkl"
scaler_path = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\scaler.pkl"
excel_path = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\Birds_info.xlsx"

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(label_encoder_path, 'rb') as file:
    label_encoder = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# Read the bird details from the Excel file
bird_details = pd.read_excel(excel_path)

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
        return render_template('index.html', error_message='No file part')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error_message='No selected file')

    if file:
        features = features_extractor(file)
        if features.size == 0:
            return render_template('index.html', error_message='Error processing audio file')

        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)
        predicted_bird = label_encoder.inverse_transform(prediction)[0]

        # Debugging prints
        print(f"Predicted Bird: {predicted_bird}")
        
        # Find bird details in the DataFrame
        bird_info = bird_details[bird_details['Bird Name'].str.strip() == predicted_bird.strip()]
        
        if bird_info.empty:
            # Debugging print to see all available bird names
            available_birds = bird_details['Bird Name'].str.strip().tolist()
            print(f"Available Bird Names: {available_birds}")
            
            return render_template('index.html', error_message=f'No information found for {predicted_bird}')

        bird_info = bird_info.iloc[0]  # Access the first row
        
        # Debugging prints
        print(f"Bird Info: {bird_info}")
        print(f"Bird Details: {bird_details}")

        # Generate the map
        latitude_range = bird_info['Latitude'].split(' to ')
        longitude_range = bird_info['Longitude'].split(' to ')
        avg_latitude = np.mean([float(lat.rstrip('°N').rstrip('°S')) for lat in latitude_range])
        avg_longitude = np.mean([float(lon.rstrip('°E').rstrip('°W')) for lon in longitude_range])

        bird_map = folium.Map(location=[avg_latitude, avg_longitude], zoom_start=4)
        folium.Marker([avg_latitude, avg_longitude], popup=predicted_bird).add_to(bird_map)
        
        # Save the map in the templates directory
        map_path = os.path.join(app.root_path, 'templates', 'map.html')
        bird_map.save(map_path)
        return render_template('result.html', bird_info=bird_info)

if __name__ == '__main__':
    app.run(debug=True)'''

from flask import Flask, request, render_template
import os
import numpy as np
import librosa
import pickle
import pandas as pd
import folium

app = Flask(__name__)

# Load the model, label encoder, and scaler
model_path = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\bird_sound_model_hybrid.pkl"
label_encoder_path = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\label_encoder.pkl"
scaler_path = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\scaler.pkl"
excel_path = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\Birds_info.xlsx"

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(label_encoder_path, 'rb') as file:
    label_encoder = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# Read the bird details from the Excel file
bird_details = pd.read_excel(excel_path)

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

@app.route('/explore')
def explore():
    birds = bird_details.to_dict(orient='records')
    return render_template('explore.html', birds=birds)

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('upload.html', error_message='No file part')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('upload.html', error_message='No selected file')

    if file:
        features = features_extractor(file)
        if features.size == 0:
            return render_template('upload.html', error_message='Error processing audio file')

        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)
        predicted_bird = label_encoder.inverse_transform(prediction)[0]

        # Debugging prints
        print(f"Predicted Bird: {predicted_bird}")
        
        # Find bird details in the DataFrame
        bird_info = bird_details[bird_details['Bird Name'].str.strip() == predicted_bird.strip()]
        
        if bird_info.empty:
            # Debugging print to see all available bird names
            available_birds = bird_details['Bird Name'].str.strip().tolist()
            print(f"Available Bird Names: {available_birds}")
            
            return render_template('upload.html', error_message=f'No information found for {predicted_bird}')

        bird_info = bird_info.iloc[0]  # Access the first row
        
        # Debugging prints
        print(f"Bird Info: {bird_info}")
        print(f"Bird Details: {bird_details}")

        # Generate the map
        latitude_range = bird_info['Latitude'].split(' to ')
        longitude_range = bird_info['Longitude'].split(' to ')
        avg_latitude = np.mean([float(lat.rstrip('°N').rstrip('°S')) for lat in latitude_range])
        avg_longitude = np.mean([float(lon.rstrip('°E').rstrip('°W')) for lon in longitude_range])

        bird_map = folium.Map(location=[avg_latitude, avg_longitude], zoom_start=4)
        folium.Marker([avg_latitude, avg_longitude], popup=predicted_bird).add_to(bird_map)
        
        # Save the map in the templates directory
        map_path = os.path.join(app.root_path, 'templates', 'map.html')
        bird_map.save(map_path)
        return render_template('result.html', bird_info=bird_info)

if __name__ == '__main__':
    app.run(debug=True)

