import webbrowser
import time
import threading
from app import app, load_model_and_labels

def open_browser():
    """Open browser after a short delay"""
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("=" * 50)
    print("Speech Emotion Recognition System")
    print("=" * 50)
    print("\nLoading model...")
    load_model_and_labels()
    print("\nStarting server...")
    print("Opening browser at http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    # Open browser in a separate thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run Flask app
    app.run(debug=False, host='127.0.0.1', port=5000)
