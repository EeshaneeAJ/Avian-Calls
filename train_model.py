import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import pickle

# Set the paths
audio_dir = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\audios"
model_save_path = "bird_sound_model_hybrid.pkl"
label_encoder_path = "label_encoder.pkl"
scaler_path = "scaler.pkl"

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

# Load and prepare the dataset
def load_data(audio_dir):
    features = []
    labels = []
    for species in os.listdir(audio_dir):
        species_dir = os.path.join(audio_dir, species)
        if os.path.isdir(species_dir):
            for audio_file in os.listdir(species_dir):
                if audio_file.endswith(".mp3") or audio_file.endswith(".wav"):  # Check for MP3 or WAV files
                    audio_path = os.path.join(species_dir, audio_file)
                    print(f"Loading: {audio_path}")
                    feature = features_extractor(audio_path)
                    if feature.shape != (0,):  # Check if feature is not empty
                        features.append(feature)
                        labels.append(species)
                    else:
                        print(f"Skipping empty feature: {audio_path}")
                else:
                    print(f"Skipping non-audio file: {audio_file}")
        else:
            print(f"Skipping non-directory: {species_dir}")
    
    print(f"Loaded {len(features)} samples.")
    return np.array(features), np.array(labels)

# Load the data
features, labels = load_data(audio_dir)

# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Save the label encoder
with open(label_encoder_path, 'wb') as file:
    pickle.dump(label_encoder, file)

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Save the scaler
with open(scaler_path, 'wb') as file:
    pickle.dump(scaler, file)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, encoded_labels, test_size=0.1, random_state=42)

# Optimize SVM hyperparameters using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

svm = SVC(probability=True)
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_svm_model = grid_search.best_estimator_

# Make predictions
y_pred = best_svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Classes')
plt.ylabel('Original Classes')
plt.show()

# Save the model
with open(model_save_path, 'wb') as file:
    pickle.dump(best_svm_model, file)


