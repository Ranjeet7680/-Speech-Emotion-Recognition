"""
Create a dummy model for demo purposes
This allows testing the web interface without training
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def create_demo_model():
    """Create a simple untrained model for demo"""
    print("Creating demo model...")
    
    # Model parameters
    input_shape = (40, 100)  # MFCC shape
    num_classes = 7
    
    # Create simple CNN model
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((*input_shape, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save model
    model.save('emotion_model.h5')
    print("✓ Model saved as emotion_model.h5")
    
    # Save emotion labels
    emotions = np.array(['angry', 'happy', 'sad', 'neutral', 'fear', 'disgust', 'surprise'])
    np.save('label_encoder.npy', emotions)
    print("✓ Labels saved as label_encoder.npy")
    
    print("\nDemo model created successfully!")
    print("Note: This is an untrained model for testing the interface.")
    print("Train a real model using train.py with your dataset for actual predictions.")

if __name__ == '__main__':
    create_demo_model()
