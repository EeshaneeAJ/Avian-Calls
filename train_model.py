'''import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import audioread
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential  # Ensure to use tensorflow.keras here
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical

# Set the paths
audio_dir = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\audios"
model_save_path = "bird_sound_model.keras"

# Function to extract features from audio files
def features_extractor(file):
    audio, sample_rate = librosa.load(file, sr=None, mono=True, res_type='kaiser_fast')
    if audio.size == 0:
        print(f"Empty audio file: {file}")
        return np.array([])  # Return empty array if audio is empty

    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
    mfccs_scaled_features = np.mean(mfccs.T, axis=0)
    return mfccs_scaled_features


# Load and prepare the dataset
def load_data(audio_dir):
    features = []
    labels = []
    for species in os.listdir(audio_dir):
        species_dir = os.path.join(audio_dir, species)
        if os.path.isdir(species_dir):
            for audio_file in os.listdir(species_dir):
                if audio_file.endswith(".mp3"):  # Check for MP3 files
                    audio_path = os.path.join(species_dir, audio_file)
                    print(f"Loading: {audio_path}")
                    feature = features_extractor(audio_path)
                    if feature.shape != (0,):  # Check if feature is not empty
                        features.append(feature)
                        labels.append(species)
                    else:
                        print(f"Skipping empty feature: {audio_path}")
                else:
                    print(f"Skipping non-MP3 file: {audio_file}")
        else:
            print(f"Skipping non-directory: {species_dir}")
    
    print(f"Loaded {len(features)} samples.")
    return np.array(features), np.array(labels)



# Load the data
features, labels = load_data(audio_dir)

# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
one_hot_labels = to_categorical(encoded_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, one_hot_labels, test_size=0.2, random_state=42)

# Reshape the data for the Conv1D model
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Build the model
model = Sequential()
model.add(Conv1D(32, kernel_size=5, activation='relu', input_shape=(80, 1)))  # Adjust input shape here
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))  # Adjust this input dimension to match the output of Flatten()
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=45, batch_size=4, validation_data=(X_test, y_test))

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
accuracy1 = model.evaluate(X_train, y_train)[1]
print(f"Accuracy on the train set: {accuracy1 * 100:.2f}%")

# Save the model
model.save(model_save_path)

# Plot confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Classes')
plt.ylabel('Original Classes')
plt.show()'''

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC

# Set the paths
audio_dir = "C:\\Users\\Abhay\\Desktop\\bird_sound_classifier\\bird_sound_classifier\\audios"
model_save_path = "bird_sound_model_svm.pkl"

# Function to extract features from audio files
def features_extractor(file):
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
    
    # Combine features
    combined_features = np.hstack((mfccs_scaled_features, chroma_scaled_features, spectral_contrast_scaled_features))
    return combined_features

# Load and prepare the dataset
def load_data(audio_dir):
    features = []
    labels = []
    for species in os.listdir(audio_dir):
        species_dir = os.path.join(audio_dir, species)
        if os.path.isdir(species_dir):
            for audio_file in os.listdir(species_dir):
                if audio_file.endswith(".mp3"):  # Check for MP3 files
                    audio_path = os.path.join(species_dir, audio_file)
                    print(f"Loading: {audio_path}")
                    feature = features_extractor(audio_path)
                    if feature.shape != (0,):  # Check if feature is not empty
                        features.append(feature)
                        labels.append(species)
                    else:
                        print(f"Skipping empty feature: {audio_path}")
                else:
                    print(f"Skipping non-MP3 file: {audio_file}")
        else:
            print(f"Skipping non-directory: {species_dir}")
    
    print(f"Loaded {len(features)} samples.")
    return np.array(features), np.array(labels)

# Load the data
features, labels = load_data(audio_dir)

# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, encoded_labels, test_size=0.2, random_state=42)

# Initialize and train the SVM classifier
svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)  # Experiment with different kernels and hyperparameters
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

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
import pickle
with open(model_save_path, 'wb') as file:
    pickle.dump(svm_model, file)
