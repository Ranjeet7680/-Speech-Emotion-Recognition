import numpy as np
from tensorflow import keras
from feature_extraction import extract_mfcc

def predict_emotion(audio_path, model_path='emotion_model.h5', 
                    label_encoder_path='label_encoder.npy'):
    """Predict emotion from audio file"""
    
    # Load model and label encoder
    model = keras.models.load_model(model_path)
    emotions = np.load(label_encoder_path, allow_pickle=True)
    
    # Extract features
    mfcc = extract_mfcc(audio_path)
    mfcc = np.expand_dims(mfcc, axis=0)
    
    # Predict
    predictions = model.predict(mfcc)
    predicted_emotion = emotions[np.argmax(predictions)]
    confidence = np.max(predictions)
    
    # Get all probabilities
    emotion_probs = {emotion: float(prob) 
                     for emotion, prob in zip(emotions, predictions[0])}
    
    return predicted_emotion, confidence, emotion_probs

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file_path>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    emotion, confidence, probs = predict_emotion(audio_file)
    
    print(f"\nPredicted Emotion: {emotion}")
    print(f"Confidence: {confidence:.2%}\n")
    print("All probabilities:")
    for emotion, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {emotion}: {prob:.2%}")
