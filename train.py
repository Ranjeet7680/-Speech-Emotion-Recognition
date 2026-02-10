import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from feature_extraction import extract_mfcc
from models import create_cnn_model, create_lstm_model

def load_dataset(data_path, emotions):
    """Load audio files and extract features"""
    features = []
    labels = []
    
    for emotion in emotions:
        emotion_path = os.path.join(data_path, emotion)
        if not os.path.exists(emotion_path):
            continue
            
        for file in os.listdir(emotion_path):
            if file.endswith('.wav'):
                file_path = os.path.join(emotion_path, file)
                try:
                    mfcc = extract_mfcc(file_path)
                    features.append(mfcc)
                    labels.append(emotion)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    return np.array(features), np.array(labels)

def train_model(data_path, model_type='cnn', epochs=50, batch_size=32):
    """Train emotion recognition model"""
    
    # Define emotions (adjust based on your dataset)
    emotions = ['angry', 'happy', 'sad', 'neutral', 'fear', 'disgust', 'surprise']
    
    print("Loading dataset...")
    X, y = load_dataset(data_path, emotions)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = keras.utils.to_categorical(y_encoded)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Input shape: {X_train.shape[1:]}")
    
    # Create model
    if model_type == 'cnn':
        model = create_cnn_model(X_train.shape[1:], len(emotions))
    elif model_type == 'lstm':
        model = create_lstm_model(X_train.shape[1:], len(emotions))
    else:
        raise ValueError("Model type must be 'cnn' or 'lstm'")
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Save model and label encoder
    model.save('emotion_model.h5')
    np.save('label_encoder.npy', le.classes_)
    
    return model, history, le

if __name__ == '__main__':
    # Update this path to your dataset location
    DATA_PATH = './data'
    
    model, history, le = train_model(
        data_path=DATA_PATH,
        model_type='cnn',
        epochs=50,
        batch_size=32
    )
