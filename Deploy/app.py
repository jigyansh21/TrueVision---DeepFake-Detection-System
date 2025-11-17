from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import imageio

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variable to store loaded model
model = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the pre-trained model with compatibility handling"""
    global model
    if model is None:
        model_path = 'models/inceptionNet_model.h5'
        if os.path.exists(model_path):
            try:
                # Method 1: Try loading with safe_mode=False (for Keras 3.x)
                try:
                    model = keras.models.load_model(model_path, compile=False, safe_mode=False)
                    print(f"[SUCCESS] Model loaded successfully from {model_path}")
                    return model
                except Exception as e1:
                    print(f"Method 1 failed: {str(e1)[:100]}...")
                
                # Method 2: Load model architecture and weights separately
                try:
                    import h5py
                    import json
                    
                    # Try to load just the weights and reconstruct
                    with h5py.File(model_path, 'r') as f:
                        # Get model config
                        if 'model_weights' in f:
                            print("Attempting to load model weights directly...")
                            # This is complex, let's try another approach
                            pass
                except Exception as e2a:
                    pass
                
                # Method 3: Create a custom GRU that handles time_major
                try:
                    from tensorflow.keras.layers import GRU, Layer
                    from tensorflow import keras as tf_keras_module
                    
                    class CompatibleGRU(GRU):
                        def __init__(self, *args, **kwargs):
                            # Remove time_major if present
                            kwargs.pop('time_major', None)
                            super().__init__(*args, **kwargs)
                    
                    # Try loading with custom objects
                    custom_objects = {
                        'GRU': CompatibleGRU,
                        'gru': CompatibleGRU,
                    }
                    
                    model = keras.models.load_model(
                        model_path, 
                        compile=False, 
                        custom_objects=custom_objects,
                        safe_mode=False
                    )
                    print(f"[SUCCESS] Model loaded with CompatibleGRU wrapper")
                    return model
                except Exception as e3:
                    print(f"Method 3 failed: {str(e3)[:100]}...")
                
                # Method 4: Try loading with tf.keras (legacy API)
                try:
                    # Use the legacy tf.keras API
                    with tf.keras.utils.custom_object_scope({'GRU': lambda *args, **kwargs: tf.keras.layers.GRU(*args, **{k: v for k, v in kwargs.items() if k != 'time_major'})}):
                        model = tf.keras.models.load_model(model_path, compile=False)
                    print(f"[SUCCESS] Model loaded using tf.keras legacy API")
                    return model
                except Exception as e4:
                    print(f"Method 4 failed: {str(e4)[:100]}...")
                
                # Method 5: Load model in a way that ignores incompatible parameters
                try:
                    # Set environment to ignore errors
                    import warnings
                    warnings.filterwarnings('ignore')
                    
                    # Try to monkey-patch GRU
                    original_gru = tf.keras.layers.GRU.__init__
                    def patched_gru_init(self, *args, **kwargs):
                        kwargs.pop('time_major', None)
                        return original_gru(self, *args, **kwargs)
                    
                    tf.keras.layers.GRU.__init__ = patched_gru_init
                    model = keras.models.load_model(model_path, compile=False, safe_mode=False)
                    tf.keras.layers.GRU.__init__ = original_gru  # Restore
                    print(f"[SUCCESS] Model loaded with patched GRU")
                    return model
                except Exception as e5:
                    print(f"Method 5 failed: {str(e5)[:100]}...")
                
                # If all methods fail, create a dummy model that returns random predictions
                print("\n[WARNING] Could not load the trained model. Using fallback mode.")
                print("The application will run but predictions will be simulated.")
                print("For real predictions, the model needs to be retrained with TensorFlow 2.16+")
                model = "FALLBACK"  # Mark as fallback mode
                
            except Exception as e:
                print(f"[ERROR] Unexpected error loading model: {e}")
                import traceback
                traceback.print_exc()
                model = "FALLBACK"
        else:
            print(f"[ERROR] Model file not found at {model_path}")
            model = "FALLBACK"
    return model

def extract_faces_from_video(video_path, max_frames=30):
    """Extract face regions from video frames using OpenCV's DNN face detector"""
    faces = []
    
    # Load OpenCV DNN face detector
    try:
        # Try to load the DNN model files
        prototxt_path = 'models/deploy.prototxt'
        model_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
        
        if not (os.path.exists(prototxt_path) and os.path.exists(model_path)):
            # Fallback to Haar Cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            use_dnn = False
        else:
            net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            use_dnn = True
    except:
        # Fallback to Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        use_dnn = False
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // max_frames)
    
    while cap.isOpened() and len(faces) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            face_detected = False
            
            if use_dnn:
                # Use DNN face detector
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                net.setInput(blob)
                detections = net.forward()
                
                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:  # Confidence threshold
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        
                        # Ensure coordinates are within frame bounds
                        startX = max(0, startX)
                        startY = max(0, startY)
                        endX = min(w, endX)
                        endY = min(h, endY)
                        
                        face = rgb_frame[startY:endY, startX:endX]
                        if face.size > 0:
                            face_resized = cv2.resize(face, (224, 224))
                            faces.append(face_resized)
                            face_detected = True
                            break
            else:
                # Use Haar Cascade
                detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(detected_faces) > 0:
                    # Use the first detected face
                    (x, y, w, h) = detected_faces[0]
                    face = rgb_frame[y:y+h, x:x+w]
                    if face.size > 0:
                        face_resized = cv2.resize(face, (224, 224))
                        faces.append(face_resized)
                        face_detected = True
        
        frame_count += 1
    
    cap.release()
    return faces

def preprocess_faces(faces):
    """Preprocess face images for model input"""
    if len(faces) == 0:
        return None
    
    # Normalize pixel values to [0, 1]
    processed = []
    for face in faces:
        face_array = np.array(face, dtype=np.float32) / 255.0
        processed.append(face_array)
    
    # Pad or truncate to fixed length (30 frames)
    max_frames = 30
    if len(processed) < max_frames:
        # Pad with last frame
        last_frame = processed[-1] if processed else np.zeros((224, 224, 3))
        while len(processed) < max_frames:
            processed.append(last_frame)
    else:
        # Truncate to max_frames
        processed = processed[:max_frames]
    
    # Convert to numpy array and add batch dimension
    processed = np.array(processed)
    processed = np.expand_dims(processed, axis=0)  # Shape: (1, 30, 224, 224, 3)
    
    return processed

def extract_cnn_features(faces, max_frames=20):
    """
    Extract CNN features from face images using InceptionV3.
    The model expects (20, 2048) feature vectors.
    """
    from tensorflow.keras.applications import InceptionV3
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    
    # Load InceptionV3 model for feature extraction
    cnn_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(224, 224, 3)
    )
    cnn_model.trainable = False
    
    # Preprocess faces for InceptionV3
    processed_faces = []
    for face in faces:
        # Resize to 224x224 if needed
        if face.shape[:2] != (224, 224):
            face_resized = cv2.resize(face, (224, 224))
        else:
            face_resized = face
        
        # Convert to RGB if needed
        if len(face_resized.shape) == 2:
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
        elif face_resized.shape[2] == 4:
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_RGBA2RGB)
        
        # Preprocess for InceptionV3 (expects values in [-1, 1] range)
        face_preprocessed = preprocess_input(face_resized.astype(np.float32))
        processed_faces.append(face_preprocessed)
    
    # Pad or truncate to 20 frames
    if len(processed_faces) < max_frames:
        # Pad with last frame
        last_frame = processed_faces[-1] if processed_faces else np.zeros((224, 224, 3))
        while len(processed_faces) < max_frames:
            processed_faces.append(last_frame)
    else:
        # Truncate to max_frames
        processed_faces = processed_faces[:max_frames]
    
    # Convert to numpy array
    processed_faces = np.array(processed_faces)  # Shape: (20, 224, 224, 3)
    
    # Extract features using CNN
    features = cnn_model.predict(processed_faces, verbose=0)  # Shape: (20, 2048)
    
    # Add batch dimension
    features = np.expand_dims(features, axis=0)  # Shape: (1, 20, 2048)
    
    # Create mask/sequence length (all ones since we have 20 frames)
    mask = np.ones((1, max_frames), dtype=np.float32)  # Shape: (1, 20)
    
    return features, mask

def predict_video(video_path):
    """Predict if video is REAL or FAKE"""
    try:
        # Load model
        model = load_model()
        if model is None or model == "FALLBACK":
            # Fallback: Return a simulated prediction based on video analysis
            print("Using fallback prediction mode...")
            try:
                # Extract faces to at least verify the video processing works
                faces = extract_faces_from_video(video_path, max_frames=10)
                if len(faces) == 0:
                    return "ERROR: No faces detected in the video. Please upload a video with clear face visibility."
                
                # Simulate a prediction (in real scenario, this would use the model)
                import random
                # For demo purposes, return a random result
                # In production, you'd need the actual model
                result = random.choice(["REAL", "FAKE"])
                print(f"Fallback prediction: {result} (Note: This is simulated, not from the trained model)")
                return f"{result} (Simulated - Model not loaded)"
            except Exception as e:
                return f"ERROR: {str(e)}"
        
        # Check if model is actually a Keras model object
        if not hasattr(model, 'predict'):
            return "ERROR: Model object is invalid. Please check the model file."
        
        # Extract faces from video
        print("Extracting faces from video...")
        faces = extract_faces_from_video(video_path, max_frames=30)
        
        if len(faces) == 0:
            return "ERROR: No faces detected in the video. Please upload a video with clear face visibility."
        
        print(f"Extracted {len(faces)} face frames")
        
        # Extract CNN features (model expects features, not raw images)
        print("Extracting CNN features...")
        try:
            features, mask = extract_cnn_features(faces, max_frames=20)
            print(f"Features shape: {features.shape}, Mask shape: {mask.shape}")
        except Exception as e:
            print(f"Error extracting features: {e}")
            return f"ERROR: Failed to extract features from video frames: {str(e)}"
        
        # Make prediction with both inputs
        print("Making prediction...")
        try:
            prediction = model.predict([features, mask], verbose=0)
        except Exception as e:
            print(f"Prediction error: {e}")
            return f"ERROR: Prediction failed: {str(e)}"
        
        # Get class prediction
        # Assuming model outputs: [probability_real, probability_fake] or [probability_fake]
        if prediction.shape[1] == 2:
            # Binary classification with 2 outputs
            predicted_class = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
        else:
            # Single output (probability of fake)
            predicted_class = 1 if prediction[0][0] > 0.5 else 0
            confidence = float(prediction[0][0]) if predicted_class == 1 else float(1 - prediction[0][0])
        
        # Map to labels: 0 = REAL, 1 = FAKE
        result = "FAKE" if predicted_class == 1 else "REAL"
        
        print(f"Prediction: {result} (Confidence: {confidence:.2%})")
        
        return result
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return f"ERROR: {str(e)}"

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('upload.html', filename=None, prediction=None)

@app.route('/', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        flash(f'Video uploaded successfully: {filename}')
        return render_template('upload.html', filename=filename, prediction=None)
    else:
        flash('Invalid file type. Please upload a video file (MP4, AVI, MOV, etc.)')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_video(filename):
    """Serve uploaded video files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict/<filename>')
def sequence_prediction(filename):
    """Analyze uploaded video and return prediction"""
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(video_path):
        flash(f'Video file not found: {filename}')
        return render_template('upload.html', filename=None, prediction=None)
    
    # Make prediction
    prediction = predict_video(video_path)
    
    return render_template('upload.html', filename=filename, prediction=prediction)

if __name__ == '__main__':
    print("=" * 50)
    print("DeepFake Detection System")
    print("=" * 50)
    print("\nLoading model...")
    load_model()
    print("\nStarting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
