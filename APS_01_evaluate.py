#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:45:18 2025

@author: adelino
"""
# from os import listdir
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import librosa
import librosa.display
import numpy as np
import dill
import keras
from scipy import signal
from scipy.io import wavfile
import math
import vad
from pathlib import Path

gender_dict = {"male": -1, "female": +1, "other": 0}
LABEL_FILE   = 'saves/label_data_file.pth'
SCALER_FILE  = 'saves/scaler_data_file.pth'
LAST_MODEL_SAVE = 'saves/last_model_file.pth'
BEST_MODEL_FILE = 'best_model.keras'
RAW_MODEL_FILE = 'saves/raw_model_file.pth'
# model.load_weights(LAST_MODEL_SAVE)

with open(LABEL_FILE, 'rb') as file:
    labels = dill.load(file)
with open(SCALER_FILE, 'rb') as file:
    scaler = dill.load(file)
with open(RAW_MODEL_FILE, 'rb') as file:
    model = dill.load(file)

print("labels: ", labels)
# model.load_weights(BEST_MODEL_FILE)
model = keras.saving.load_model(BEST_MODEL_FILE)
# -----------------------------------------------------------------------------
def app_feature_extraction(path, gender, sampling_rate = 8000):
    features = list()
    audio, _ = librosa.load(path, sr=sampling_rate)
    
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
def my_voice_prediction(path, gender, model, scaler, test_number):
    features = app_feature_extraction(path, gender)
    gender = features[0]
    features = scaler.transform(features.reshape(1, -1))  # reshape because we have a single sample
    features = features[0]   # beacause the shape is (1, 24), but we want (24, ) as shape
    features[0] = gender     # in this way the gender will be always +1, 0 or -1
    prediction = model.predict(np.expand_dims(features, axis=0))
    idxSort = np.argsort(prediction)[0]
    prediction = prediction[0]
    print("predicted age of test {:02}: {:}, score: {:0.4f}".format(test_number, labels[idxSort[-1]],prediction[idxSort[-1]])) 
# =============================================================================
def build_folders_to_save(file_name):
    split_path = file_name.split('/')
    if (split_path[0] == ''):
        split_path = split_path[1:]
    for idx in range(1,len(split_path)):
        curDir = '/'.join(split_path[:idx])
        if (not os.path.exists(curDir)):
            os.mkdir(curDir)
# =============================================================================
def list_contend(folder='./', pattern=(), size_limit=None):   
    if (len(pattern) > 0):
        pattern = tuple([x.upper() for x in pattern])
        list_files = [os.path.join(root, name)
                 for root, dirs, files in os.walk(folder)
                     for name in files
                         if name.upper().endswith(pattern)]
    else:
        list_files = [os.path.join(root, name)
                 for root, dirs, files in os.walk(folder)
                     for name in files]
    if not (size_limit == None):
        temp_list_files = [nfile for nfile in list_files if (os.path.getsize(nfile) < size_limit)]
        list_files  = temp_list_files
    list_files.sort()
    return list_files
# =============================================================================
def vadFrame2SampleBase(audio,sr,vad,win_length=0.025,hop_length=0.01):
    nData = len(audio)
    nVAD = len(vad)
    sbvad = np.zeros((nData,))
    nWlen = int(win_length*sr)
    nHlen = int(hop_length*sr)
    k = 0
    for idx in range(0,nData,nHlen):
        if (k < nVAD) and (vad[k] == 1):
            sbvad[idx:(idx+nWlen)] = 1
        k += 1
    return (sbvad == 1).nonzero()[0], (sbvad == 0).nonzero()[0]
# =============================================================================
def sub_sample_audio(AUDIO_IN, AUDIO_OUT,base_name,subSize,rate,nSecBySubSample = 10):
    lst_FILES = list_contend(AUDIO_IN, pattern=(".wav",".mp3",))
    b, a = signal.butter(4, [80/rate,(rate-80)/rate], btype='bandpass')
    for file_name in lst_FILES:
        sr, clip = wavfile.read(file_name)
        #fullTime = len(clip)/sr
        # --- extrai apenas as voz
        k = 1
        if (clip.dtype == np.int16):
            k = 32767
        if ((clip.dtype == np.int32) or (clip.dtype == np.float32)):
            k = 2147483647
        if (clip.dtype == np.uint8):
            k = 255
        clip = clip + np.random.normal(0, 1/(k*3), len(clip))
        n_FFT = 2 ** math.ceil(math.log2(0.025*sr))
        vad_sohn = vad.VAD(clip, sr, nFFT=n_FFT)
        vadIDX, noiseIDX = vadFrame2SampleBase(clip,sr,vad_sohn)            
        #noiseBass = np.std(clip[noiseIDX])/int16_max
        clip = clip[vadIDX]
        n_samples = round(len(clip) * float(rate) / sr)
        clip = signal.resample(clip, n_samples)
        clip = signal.filtfilt(b, a, clip)
        vadTime = len(clip)/rate
        nSubSample = int(vadTime/subSize)
        nPtsBySubSample = nSecBySubSample*rate
        file_base = Path(file_name).stem
        for i in range (0,nSubSample):
            idxIni = int(i*nPtsBySubSample)
            if (i + 1) < nSubSample:
                idxFim = int((i+1)*nPtsBySubSample)
            else:
                idxFim = int(vadTime*rate)
            sub_sample_filename = AUDIO_OUT + '{:}_{:}_{:03d}.wav'.format(file_base,base_name,i)
            build_folders_to_save(sub_sample_filename)
            wavfile.write(sub_sample_filename, rate,np.array(clip[idxIni:idxFim],dtype=np.int16))
# =============================================================================
nSecBySubSample = 10
sb_rate = 8000
QST_FOLDER = 'eval_files/'
QST_SUBSAMPLE_DATA = './eval_files/QST/'
BASENAME = 'Sub_Sample'
sub_sample_audio(QST_FOLDER,QST_SUBSAMPLE_DATA,BASENAME,nSecBySubSample,sb_rate,nSecBySubSample)
file_list = list_contend(QST_SUBSAMPLE_DATA,('.wav',))

gender = gender_dict["male"]

for idx, filename in enumerate(file_list):
    my_voice_prediction(filename, gender, model, scaler, test_number=idx)
