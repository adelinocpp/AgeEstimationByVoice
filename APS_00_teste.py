#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:14:13 2025

@author: adelino
"""

# from os import listdir
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from os.path import join
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sys
import csv
from sklearn.preprocessing import LabelEncoder, StandardScaler
import dill
from sklearn.model_selection import train_test_split
import keras
from keras import models
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
# import sklearn
import subprocess
from time import sleep

gender_dict = {"male": -1, "female": +1, "other": 0}

MCV_PT_PATH = "/media/CAEMLYN/30_Pesquisa/Mozzila_Common_Voice/cv-corpus-17.0-2024-03-15/pt"

TSV_FILES = [#"clip_durations.tsv",
"dev.tsv",
"invalidated.tsv",
"other.tsv",
#"reported.tsv",
"test.tsv",
"train.tsv",
#"unvalidated_sentences.tsv",
"validated.tsv"]
#"validated_sentences.tsv"]

# =============================================================================
def read_encode_audio_phone_channel(file_name,enctype='OPUS',sr=8000,bitrate=12.2,perLoss=0):
    file_save = "./temp.wav"
    max_count = 20
    if (enctype.upper() == 'GSM'):
        fileCODE = 'temp.gsm'
        encodeString = "sox {:} -r {:} -t gsm {:}".format(file_name,sr,fileCODE);
        dencodeString = "sox {:} -e signed-integer -b 16 -r {:} {:}".format(fileCODE,sr,file_save)
    else:
        fileCODE = 'temp.opus'
        # encodeString = "opusenc {:} --bitrate {:5.3f} {:}".format(file_name,bitrate,fileCODE)
        encodeString = "/home/adelino/miniconda3/bin/ffmpeg -hide_banner -loglevel quiet -i {:} -r {:} -f wav - | opusenc --bitrate {:5.3f} - --quiet --vbr {:}".format(file_name,sr,bitrate,fileCODE)
        dencodeString = "opusdec {:} --packet-loss {:03d} --rate {:} {:} --quiet".format(fileCODE,perLoss,sr,file_save)
    
    deleteCODE = "rm {:}".format(fileCODE)
    subprocess.Popen(encodeString, shell=True, stdout=subprocess.PIPE).wait()
    count = 0
    while (not os.path.exists(fileCODE)) or (count < max_count):
        sleep(0.0001)
        count+=1
        
    subprocess.Popen(dencodeString, shell=True, stdout=subprocess.PIPE).wait()
    count = 0
    while (not os.path.exists(file_save)) or (count < max_count):
        sleep(0.0001)
        count+=1
    
    subprocess.Popen(deleteCODE, shell=True, stdout=subprocess.PIPE).wait()
    audio, sr_read = librosa.load(file_save, sr=sr)
    
    deleteCODE = "rm {:}".format(file_save)
    subprocess.Popen(deleteCODE, shell=True, stdout=subprocess.PIPE).wait()
    return audio, sr_read 
# =============================================================================
def get_max_length():
    dfTemp = pd.read_csv(join(MCV_PT_PATH,"clip_durations.tsv"),sep='\t')
    return dfTemp["duration[ms]"].max()
# -----------------------------------------------------------------------------
def get_dataframe(path):
    df = pd.read_csv(path,sep='\t')
    if ("age" in df.columns):
        return df[pd.notna(df['age'])]
    else:
        return pd.DataFrame([])
# -----------------------------------------------------------------------------
def get_mp3_name(path):
    return "{}.mp3".format(path[:-4].split("clips/")[-1])
# -----------------------------------------------------------------------------
def get_age(df, path):
    path = get_mp3_name(path)
    return df.loc[df['path'] == path]["age"].values[0]
# -----------------------------------------------------------------------------
def get_gender(df, path):
    path = get_mp3_name(path)
    try:
        gender = df.loc[df['path'] == path]["gender"].values[0]
        return gender_dict[gender]
    except:
        return gender_dict["other"]
# -----------------------------------------------------------------------------
def get_spectrogram(path, sampling_rate = 8000, display = True):
    # Load an audio file as a floating point time series.
    audio , _ = librosa.load(path, sr=sampling_rate)

    # Short-time Fourier transform (STFT).
    stft = abs(librosa.stft(audio))

    # Convert an amplitude spectrogram to dB-scaled spectrogram.
    spectrogram = librosa.amplitude_to_db(stft)

    if display:
        plt.figure(figsize=(9, 3))
        librosa.display.specshow(spectrogram, sr=sampling_rate, x_axis='time', y_axis='log')
        plt.colorbar()
    return spectrogram
# -----------------------------------------------------------------------------
def feature_extraction(path, feat_type = 'RAW', sampling_rate = 8000):
    features = list()
    
    if (feat_type == 'OPUS'):
        audio, sr_read = read_encode_audio_phone_channel(path,'OPUS',sr=sampling_rate)
    if (feat_type == 'GSM'):
        audio, sr_read = read_encode_audio_phone_channel(path,'GSM',sr=sampling_rate)
    if (not (feat_type == 'OPUS')) and (not (feat_type == 'GSM')):
        audio, sr_read = librosa.load(path, sr=sampling_rate)

    gender = get_gender(df, path)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sampling_rate))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sampling_rate))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sampling_rate))
    features.append(gender)
    features.append(spectral_centroid)
    features.append(spectral_bandwidth)
    features.append(spectral_rolloff)
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate)
    for el in mfcc:
        features.append(np.mean(el))
    
    return np.asarray(features, dtype=float)
# -----------------------------------------------------------------------------
def get_wav_name(path):
    info = path.split("/")
    filename = join("cv_corpus_v1", "wav-files", info[0], "{}.wav".format(info[1][:-4]))
    return filename
# -----------------------------------------------------------------------------
def parse(row):
    features = feature_extraction(get_wav_name(row["filename"]))
    label = row["age"]
    return [features, label]
# -----------------------------------------------------------------------------
def create_header():
    header = 'filename gender spectral_centroid spectral_bandwidth spectral_rolloff'
    for i in range(1, 21):
        header += ' mfcc{}'.format(i)
    header += ' label'
    header = header.split()
    return header
# -----------------------------------------------------------------------------
def create_feature_csv(csv_name, header):
    vecFeatureTypes = ['RAW', 'GSM','OPUS']
    df = get_dataframe(join(MCV_PT_PATH, csv_name))
    print("Calculando {:}".format(csv_name))
    if (df.shape[0] > 0):
        with open(join("./", "feature-csv", csv_name), 'w') as file:
            writer = csv.writer(file)
            writer.writerow(header)
        for index, row in df.iterrows():
            FILE_NAME = join(MCV_PT_PATH, "clips",row["path"])
            if (os.path.exists(FILE_NAME)):
                for feat_types in vecFeatureTypes:
                    to_append = list()
                    filename = row["path"]
                    features = feature_extraction(FILE_NAME, feat_type=feat_types)
                    label = row["age"]
                    to_append.append(filename)
                    to_append.extend(features)
                    to_append.append(label)
                    with open(join("./", "feature-csv", csv_name), 'a') as file:
                        writer = csv.writer(file)
                        writer.writerow(to_append)
            else:
                continue
    else:
        return
# -----------------------------------------------------------------------------
def get_labels(data):
    labels = data.iloc[:, -1]
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    return labels, encoder
# -----------------------------------------------------------------------------
def scale_features(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(np.array(data.iloc[:, 1:-1], dtype = float))
    # with data.iloc[:, 1:-1] we don't consider filename and label

    # in this way, gender will be always -1, 0 or +1 and so it won't be scaled
    for i in range(len(scaled_data)):
        scaled_data[i][0] = data.iloc[i, 1]
        
    return scaled_data, scaler
# -----------------------------------------------------------------------------
    
max_length = get_max_length()
max_length_set = 695060352

df = get_dataframe(join(MCV_PT_PATH, "other.tsv"))
 
FILE_TEST = df["path"].iloc[0]

print("Age ({:}): {:}.".format(FILE_TEST,get_age(df, join(MCV_PT_PATH,"clips", FILE_TEST))))
print("Gender ({:}): {:}.".format(FILE_TEST,get_gender(df, join(MCV_PT_PATH,"clips", FILE_TEST))))

SPECTROGRAM = False
if (SPECTROGRAM):
    _ = get_spectrogram(join(MCV_PT_PATH,"clips", FILE_TEST))

features = feature_extraction(join(MCV_PT_PATH, "clips", FILE_TEST))
print("features: ", features)
print("shape: ", features.shape)

MAKE_FEATURES_FILES = True

if (MAKE_FEATURES_FILES):
    header = create_header()
    for el in TSV_FILES:
        create_feature_csv(el, header)
    
RAW_DATA_FILE = 'saves/raw_features_data_file.pth'
MAKE_RAW_DATA_FILE = True
if (MAKE_RAW_DATA_FILE):
    data = pd.concat([pd.read_csv(join("./", "feature-csv", "dev.tsv")), \
                      pd.read_csv(join("./", "feature-csv", "invalidated.tsv")), \
                      pd.read_csv(join("./", "feature-csv", "other.tsv")), \
                      # pd.read_csv(join("./", "feature-csv", "reported.tsv")), \
                      pd.read_csv(join("./", "feature-csv", "test.tsv")), \
                      pd.read_csv(join("./", "feature-csv", "train.tsv")), \
                      # pd.read_csv(join("./", "feature-csv", "unvalidated_sentences.tsv")), \
                      pd.read_csv(join("./", "feature-csv", "validated.tsv")), \
                      # pd.read_csv(join("./", "feature-csv", "validated_sentences.tsv")) 
                      ])
    with open(RAW_DATA_FILE, 'wb') as file:
        dill.dump(data, file)    
else:
    with open(RAW_DATA_FILE, 'rb') as file:
        data = dill.load(file)

print(data.shape)
data.head()


X_TRAIN_FILE = 'saves/x_train_data_file.pth'
Y_TRAIN_FILE = 'saves/y_train_data_file.pth'
X_TEST_FILE  = 'saves/x_test_data_file.pth'
Y_TEST_FILE  = 'saves/y_test_data_file.pth'
X_VAL_FILE   = 'saves/x_val_data_file.pth'
Y_VAL_FILE   = 'saves/y_val_data_file.pth'
LABEL_FILE   = 'saves/label_data_file.pth'
SCALER_FILE  = 'saves/scaler_data_file.pth'
SPLIT_DATA = True
if (SPLIT_DATA):
    y, encoder = get_labels(data)
    labels = encoder.classes_
    x, scaler = scale_features(data)
    with open(LABEL_FILE, 'wb') as file:
        dill.dump(labels, file) 
    with open(SCALER_FILE, 'wb') as file:
        dill.dump(scaler, file) 
    # balanced split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=0, stratify=y_test)
    with open(X_TRAIN_FILE, 'wb') as file:
        dill.dump(x_train, file) 
    with open(Y_TRAIN_FILE, 'wb') as file:
        dill.dump(y_train, file) 
    with open(X_TEST_FILE, 'wb') as file:
        dill.dump(x_test, file) 
    with open(Y_TEST_FILE, 'wb') as file:
        dill.dump(y_test, file) 
    with open(X_VAL_FILE, 'wb') as file:
        dill.dump(x_val, file) 
    with open(Y_VAL_FILE, 'wb') as file:
        dill.dump(y_val, file) 
else:
    with open(LABEL_FILE, 'rb') as file:
        labels = dill.load(file)
    with open(SCALER_FILE, 'rb') as file:
        scaler = dill.load(file)
    with open(X_TRAIN_FILE, 'rb') as file:
        x_train = dill.load(file)
    with open(Y_TRAIN_FILE, 'rb') as file:
        y_train = dill.load(file)
    with open(X_TEST_FILE, 'rb') as file:
        x_test = dill.load(file)
    with open(Y_TEST_FILE, 'rb') as file:
        y_test = dill.load(file)
    with open(X_VAL_FILE, 'rb') as file:
        x_val = dill.load(file)
    with open(Y_VAL_FILE, 'rb') as file:
        y_val = dill.load(file)
        
print("labels: ", labels)
print("x_train: ", x_train.shape)
print("x_test: ", x_test.shape)
print("x_val: ", x_val.shape)

MAKE_RAW_MODEL = True
RAW_MODEL_FILE = 'saves/raw_model_file.pth'
model = models.Sequential()
if (MAKE_RAW_MODEL):
    model.add(layers.BatchNormalization(input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(1024, activation='relu'))
    
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1024, activation='relu'))
    
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1024, activation='relu'))
    
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1024, activation='relu'))
    
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1024, activation='relu'))
    
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1024, activation='relu'))
    
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1024, activation='relu'))
    
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1024, activation='relu'))
    
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(labels.shape[0], activation='softmax'))    # note that 8 is the number of possible labels
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    with open(RAW_MODEL_FILE, 'wb') as file:
        dill.dump(model, file) 
else:
    with open(RAW_MODEL_FILE, 'rb') as file:
        model = dill.load(file)
            
print(model.summary())

BEST_MODEL_FILE = 'best_model.keras'
CHECKPOINT_FILE = 'saves/checkpoint.pth'
EARLY_STOP_FILE = 'saves/early_stop.pth'
NEW_CHECKPOINT = True
if (NEW_CHECKPOINT):
    checkpointer = ModelCheckpoint(filepath=BEST_MODEL_FILE, 
                                   verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
    
    with open(CHECKPOINT_FILE, 'wb') as file:
        dill.dump(checkpointer, file) 
    with open(EARLY_STOP_FILE, 'wb') as file:
        dill.dump(early_stopping, file) 
else:
    with open(CHECKPOINT_FILE, 'rb') as file:
        checkpointer = dill.load(file)
    with open(EARLY_STOP_FILE, 'rb') as file:
        early_stopping = dill.load(file)

vecBatchSize = [128, 256, 512,1024, 2048]
stages = ['First', 'Second', 'Third', 'Fourth', 'Fifth']
epochs = 50
acc_to_plot = []
val_acc_to_plot = []
loss_to_plot = []
val_loss_to_plot = []
TRAIN_MODEL = True
LAST_MODEL_SAVE = 'saves/last_model_file.keras'
if (TRAIN_MODEL):
    for idx, batch_size in enumerate(vecBatchSize):
        print("{:} stage trainning...".format(stages[idx]))
        history = model.fit(x_train,
                            y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[checkpointer, early_stopping],
                            validation_data=(x_val, y_val))
    
        model.load_weights(BEST_MODEL_FILE)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print("{:02d} Acuracia do teste: {:.2f} %".format(idx,test_acc*100))
        
        last_good_epoch = early_stopping.stopped_epoch - early_stopping.patience + 1
        
        acc_to_plot += history.history['accuracy'][:last_good_epoch]
        val_acc_to_plot += history.history['val_accuracy'][:last_good_epoch]
        
        filenameACC = '{:02d}_Accuracy.png'.format(idx)
        plt.figure(figsize=[10,7.5])
        plt.plot(acc_to_plot,'ro-.',linewidth=1.0)
        plt.plot(val_acc_to_plot,'bx-.',linewidth=1.0)
        plt.legend(['Acuracia de treinamento', 'Validaçao treinamento'],fontsize=14)
        plt.xlabel('Epocas ',fontsize=12)
        plt.ylabel('Acuracia',fontsize=12)
        plt.title('Curvas de acuracia',fontsize=16)
        plt.grid(color='grey', linestyle='-.', linewidth=0.5)
        plt.savefig(filenameACC,bbox_inches='tight',dpi=200)
        
        loss_to_plot += history.history['loss'][:last_good_epoch]
        val_loss_to_plot += history.history['val_loss'][:last_good_epoch]
        
        filenameLoss = '{:02d}_Loss.png'.format(idx)
        plt.figure(figsize=[10,7.5])
        plt.plot(loss_to_plot,'ro-.',linewidth=1.0)
        plt.plot(val_loss_to_plot,'bx-.',linewidth=1.0)
        plt.legend(['Perda de treinamento', 'Perda de validaçao'],fontsize=14)
        plt.xlabel('Epocas',fontsize=12)
        plt.ylabel('Curvas de Perda',fontsize=12)
        # plt.title('Loss Curves',fontsize=16)
        plt.grid(color='grey', linestyle='-.', linewidth=0.5)
        plt.savefig(filenameLoss,bbox_inches='tight',dpi=200)
        
    model.save(LAST_MODEL_SAVE)
else:
    model = keras.saving.load_model(LAST_MODEL_SAVE)


