import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def process_audio_files(dir_path):
    features = []
    labels = []
    
    for folder_name in os.listdir(dir_path):
        for subfolder_name in os.listdir(os.path.join(dir_path, folder_name)):
            # Solo procesar archivos .wav
            for file_name in os.listdir(os.path.join(dir_path, folder_name, subfolder_name)):
                if file_name.lower().endswith('.wav'):
                    y, sr = librosa.load(os.path.join(dir_path, folder_name, subfolder_name, file_name))

                    # Extraer el MFCC del audio
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
                    mfcc_mean = np.mean(mfcc, axis=1) # Se calcula el promedio para reducir la dimensión a 128

                    # Añadir el MFCC y la etiqueta a nuestras listas
                    features.append(mfcc_mean.tolist())
                    labels.append(subfolder_name)

    # Convertir a DataFrame y guardar en un archivo .csv
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv('dataset.csv', index=False)

def split_dataset(test_size=0.3):
    df = pd.read_csv('dataset.csv')
    train, test = train_test_split(df, test_size=test_size, random_state=42)
    train.to_csv('training2.csv', index=False)
    test.to_csv('testing2.csv', index=False)

    
dir_path = './sound_dataset'
process_audio_files(dir_path)
split_dataset()
