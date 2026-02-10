# Speech Emotion Recognition

Deep learning system for recognizing emotions from speech audio using MFCC features and CNN/LSTM models.

## Features

- MFCC feature extraction with librosa
- Multiple model architectures (CNN, LSTM, CNN-LSTM hybrid)
- Support for datasets: RAVDESS, TESS, EMO-DB
- Real-time emotion prediction

## Quick Start (Windows)

1. Double-click `START_HERE.bat`
2. Browser will open automatically at http://localhost:5000

## Manual Installation

```bash
pip install -r requirements.txt
```

## Create Demo Model (for testing)

```bash
python create_demo_model.py
```

## Dataset Structure

Organize your audio files in the following structure:

```
data/
├── angry/
│   ├── audio1.wav
│   └── audio2.wav
├── happy/
│   ├── audio1.wav
│   └── audio2.wav
├── sad/
└── ...
```

## Usage

### Training

```bash
python train.py
```

Modify `DATA_PATH` in `train.py` to point to your dataset location.

### Prediction

**Command Line:**
```bash
python predict.py path/to/audio.wav
```

**Web Interface:**
```bash
python app.py
```
Then open http://localhost:5000 in your browser

## Supported Emotions

- Angry
- Happy
- Sad
- Neutral
- Fear
- Disgust
- Surprise

## Model Architectures

- **CNN**: Convolutional layers for spatial feature extraction
- **LSTM**: Recurrent layers for temporal patterns
- **CNN-LSTM**: Hybrid approach combining both

## Datasets

- **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song
- **TESS**: Toronto Emotional Speech Set
- **EMO-DB**: Berlin Database of Emotional Speech
