import librosa
import numpy as np

def extract_mfcc(audio_path, n_mfcc=40, max_len=100):
    """Extract MFCC features from audio file"""
    audio, sr = librosa.load(audio_path, duration=3, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Pad or truncate to fixed length
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc

def extract_features(audio_path):
    """Extract multiple audio features"""
    audio, sr = librosa.load(audio_path, duration=3, sr=22050)
    
    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    
    # Chroma
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    
    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    
    # Combine features
    features = np.hstack([mfcc, chroma, mel])
    
    return features
