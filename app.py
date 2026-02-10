from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow import keras
from feature_extraction import extract_mfcc

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and labels
model = None
emotions = None

def load_model_and_labels():
    global model, emotions
    try:
        model = keras.models.load_model('emotion_model.h5')
        emotions = np.load('label_encoder.npy', allow_pickle=True)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model - {e}")
        print("Please train the model first using train.py")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Use WAV, MP3, OGG, or FLAC'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract features and predict
        mfcc = extract_mfcc(filepath)
        mfcc = np.expand_dims(mfcc, axis=0)
        
        predictions = model.predict(mfcc, verbose=0)
        predicted_emotion = emotions[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        
        # Get all probabilities
        emotion_probs = {emotion: float(prob) 
                        for emotion, prob in zip(emotions, predictions[0])}
        
        # Sort by probability
        sorted_probs = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'emotion': predicted_emotion,
            'confidence': confidence,
            'probabilities': dict(sorted_probs)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    load_model_and_labels()
    app.run(debug=True, host='0.0.0.0', port=5000)
