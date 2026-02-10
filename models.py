import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_cnn_model(input_shape, num_classes):
    """CNN model for emotion recognition"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((*input_shape, 1)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_lstm_model(input_shape, num_classes):
    """LSTM model for emotion recognition"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        
        layers.LSTM(64),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_cnn_lstm_model(input_shape, num_classes):
    """Hybrid CNN-LSTM model"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((*input_shape, 1)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Reshape((input_shape[0] // 2, -1)),
        
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
