"""
AI Crowd Monitoring System - Backend
Main server handling video processing, AI inference, and real-time data streaming
"""

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import threading
import time
from collections import deque
from datetime import datetime
import base64
from io import BytesIO

# Optional imports - install as needed
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLO not installed. Install with: pip install ultralytics")

try:
    import face_recognition
    import dlib
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠️  face_recognition not installed. Install with: pip install face-recognition")

# ============================================================================
# Flask App Configuration
# ============================================================================

app = Flask(__name__)
CORS(app)

# Configuration
CONFIG = {
    'VIDEO_UPLOAD_FOLDER': 'uploads/',
    'MAX_FRAME_BUFFER': 30,
    'DETECTION_THRESHOLD': 0.5,
    'HEAT_MAP_GRID_SIZE': 40,
    'MAX_FACES_STORED': 50,
}

# ============================================================================
# Global State
# ============================================================================

class SystemState:
    def __init__(self):
        self.people_count = 0
        self.total_count = 0
        self.fps = 0
        self.anomalies = 0
        self.heatmap_data = []
        self.scatter_points = deque(maxlen=100)
        self.detected_faces = deque(maxlen=CONFIG['MAX_FACES_STORED'])
        self.alerts = deque(maxlen=50)
        self.current_frame = None
        self.processing = False
        self.threshold = CONFIG['DETECTION_THRESHOLD']
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()

system_state = SystemState()

# ============================================================================
# Model Loading
# ============================================================================

def load_models():
    """Load AI models"""
    models = {}
    
    if YOLO_AVAILABLE:
        try:
            models['yolo'] = YOLO('yolov8n.pt')  # Nano model for speed
            print("✅ YOLO model loaded")
        except Exception as e:
            print(f"❌ Failed to load YOLO: {e}")
    
    return models

models = load_models()

# ============================================================================
# Detection Functions
# ============================================================================

def detect_people(frame, model, threshold=0.5):
    """
    Detect people in frame using YOLO
    Returns: count, detections list, annotated frame
    """
    if not YOLO_AVAILABLE or 'yolo' not in models:
        return 0, [], frame
    
    try:
        # Run inference
        results = models['yolo'](frame, verbose=False)
        
        detections = []
        people_count = 0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Check if detection is "person" class (class 0 in COCO)
                if box.cls == 0 and box.conf >= threshold:
                    people_count += 1
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf.cpu().numpy())
                    
                    # Center point for scatter plot
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': (cx, cy),
                        'confidence': conf,
                    })
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return people_count, detections, frame
    
    except Exception as e:
        print(f"Detection error: {e}")
        return 0, [], frame

def extract_faces(frame, detections):
    """
    Extract face regions from detections
    Returns: list of face images with confidence
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return []
    
    faces = []
    try:
        # Convert to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find faces
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            face_image = frame[top:bottom, left:right]
            faces.append({
                'image': face_image,
                'encoding': encoding,
                'location': (top, right, bottom, left),
                'confidence': 0.95,  # Placeholder
            })
    
    except Exception as e:
        print(f"Face extraction error: {e}")
    
    return faces

def recognize_faces(frame, faces, known_faces_db=None):
    """
    Recognize and identify faces
    Returns: list of identified faces
    """
    if not faces or not FACE_RECOGNITION_AVAILABLE:
        return []
    
    recognized = []
    try:
        for face in faces:
            # Simple matching with known faces
            # In production, use a proper face database
            top, right, bottom, left = face['location']
            
            recognized.append({
                'id': f"Person_{len(recognized)}",
                'location': (left, top, right, bottom),
                'confidence': face['confidence'],
                'timestamp': time.time(),
                'is_suspicious': False,  # Add your logic here
            })
    
    except Exception as e:
        print(f"Face recognition error: {e}")
    
    return recognized

# ============================================================================
# Data Processing Functions
# ============================================================================

def generate_heatmap_data(detections, frame_shape):
    """
    Generate heatmap from detection centers
    Returns: list of heatmap points
    """
    if not detections:
        return []
    
    # Create grid
    h, w = frame_shape[:2]
    grid_size = CONFIG['HEAT_MAP_GRID_SIZE']
    
    # Create density map
    density_map = {}
    max_density = 0
    
    for det in detections:
        cx, cy = det['center']
        grid_x = (cx // grid_size) * grid_size
        grid_y = (cy // grid_size) * grid_size
        key = (grid_x, grid_y)
        density_map[key] = density_map.get(key, 0) + 1
        max_density = max(max_density, density_map[key])
    
    # Convert to heatmap points
    heatmap_data = []
    for (x, y), count in density_map.items():
        intensity = min(count / max(max_density, 1), 1.0)
        heatmap_data.append({
            'x': x + grid_size // 2,
            'y': y + grid_size // 2,
            'intensity': intensity,
            'count': count,
        })
    
    return heatmap_data

def generate_scatter_points(detections):
    """
    Generate scatter plot points from detections
    Returns: list of points for scatter chart
    """
    return [{'x': det['center'][0], 'y': det['center'][1]} 
            for det in detections]

def detect_anomalies(detections, frame_shape):
    """
    Detect anomalies (crowding, sudden changes)
    Returns: count of anomalies detected
    """
    count = len(detections)
    anomalies = 0
    
    # Simple anomaly detection rules
    h, w = frame_shape[:2]
    
    # High density detection
    if count > 100:
        anomalies += 1
    
    # Cluster detection (crowding in small area)
    if count > 10:
        centers = [det['center'] for det in detections]
        # Calculate density
        for cx, cy in centers:
            nearby = sum(1 for x, y in centers 
                        if abs(x - cx) < 50 and abs(y - cy) < 50)
            if nearby > 15:
                anomalies += 1
                break
    
    return min(anomalies, 5)

# ============================================================================
# Video Capture and Processing
# ============================================================================

class VideoCapture:
    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self.lock = threading.Lock()
        self.is_running = False
    
    def start(self):
        """Start video capture"""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise Exception(f"Cannot open video source: {self.source}")
        
        self.is_running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop video capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
    
    def _process_loop(self):
        """Main processing loop"""
        frame_count = 0
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Resize for faster processing
            frame = cv2.resize(frame, (640, 480))
            
            # Run detections
            people_count, detections, annotated = detect_people(
                frame, models.get('yolo'), system_state.threshold
            )
            
            # Generate visualizations
            heatmap_data = generate_heatmap_data(detections, frame.shape)
            scatter_points = generate_scatter_points(detections)
            anomalies = detect_anomalies(detections, frame.shape)
            
            # Extract and recognize faces
            faces = extract_faces(frame, detections)
            recognized_faces = recognize_faces(frame, faces)
            
            # Update system state
            with self.lock:
                system_state.people_count = people_count
                system_state.total_count += people_count
                system_state.heatmap_data = heatmap_data
                system_state.scatter_points.extend(scatter_points)
                system_state.anomalies = anomalies
                system_state.current_frame = annotated.copy()
                
                # Update FPS
                current_time = time.time()
                system_state.frame_times.append(current_time)
                if len(system_state.frame_times) > 1:
                    fps = len(system_state.frame_times) / (
                        system_state.frame_times[-1] - system_state.frame_times[0]
                    )
                    system_state.fps = fps
            
            # Check for anomalies and generate alerts
            if anomalies > 0:
                add_alert(f"⚠️ Anomaly detected! ({anomalies} issues)", 'warning')

video_capture = VideoCapture(0)  # Default to webcam

# ============================================================================
# Alert Management
# ============================================================================

def add_alert(message, alert_type='info'):
    """Add alert to system"""
    alert = {
        'message': message,
        'type': alert_type,
        'timestamp': datetime.now().isoformat(),
    }
    system_state.alerts.append(alert)

# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    """Serve main HTML"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream video frames"""
    def generate():
        while True:
            frame = system_state.current_frame
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       buffer.tobytes() + b'\r\n')
            
            time.sleep(0.01)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detection-data')
def get_detection_data():
    """Return current detection data as JSON"""
    alerts = []
    if system_state.alerts:
        recent_alerts = list(system_state.alerts)[-5:]
        alerts = [{'message': a['message'], 'type': a['type']} 
                 for a in recent_alerts]
    
    # Prepare faces data
    faces_data = []
    if system_state.detected_faces:
        for face in list(system_state.detected_faces)[-CONFIG['MAX_FACES_STORED']:]:
            face_copy = face.copy()
            if 'image' in face_copy:
                # Encode image to base64
                _, buffer = cv2.imencode('.jpg', face_copy['image'])
                face_copy['image_base64'] = 'data:image/jpeg;base64,' + base64.b64encode(
                    buffer
                ).decode()
                del face_copy['image']
            faces_data.append(face_copy)
    
    return jsonify({
        'people_count': system_state.people_count,
        'total_count': system_state.total_count,
        'fps': system_state.fps,
        'anomalies': system_state.anomalies,
        'heatmap_data': system_state.heatmap_data,
        'scatter_points': list(system_state.scatter_points),
        'detected_faces': faces_data,
        'new_alerts': alerts,
    })

@app.route('/api/command', methods=['POST'])
def handle_command():
    """Handle commands from frontend"""
    data = request.json
    action = data.get('action')
    
    if action == 'update_threshold':
        system_state.threshold = data.get('value', 0.5)
        return jsonify({'status': 'ok', 'threshold': system_state.threshold})
    
    elif action == 'reset':
        system_state.total_count = 0
        system_state.anomalies = 0
        system_state.alerts.clear()
        return jsonify({'status': 'reset'})
    
    return jsonify({'status': 'unknown'})

@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    """Handle video file upload"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    file = request.files['video']
    filepath = f"uploads/{file.filename}"
    
    try:
        file.save(filepath)
        # Switch video source to uploaded file
        video_capture.stop()
        video_capture.source = filepath
        video_capture.start()
        
        return jsonify({'status': 'uploaded', 'filename': file.filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'running': video_capture.is_running,
        'people_detected': system_state.people_count,
        'fps': system_state.fps,
        'models_loaded': {
            'yolo': YOLO_AVAILABLE and 'yolo' in models,
            'face_recognition': FACE_RECOGNITION_AVAILABLE,
        }
    })

# ============================================================================
# Startup and Shutdown
# ============================================================================

@app.before_request
def startup():
    """Initialize on first request"""
    if not video_capture.is_running:
        try:
            video_capture.start()
            print("✅ Video capture started")
        except Exception as e:
            print(f"❌ Video capture error: {e}")

@app.teardown_appcontext
def shutdown(exception=None):
    """Cleanup on shutdown"""
    video_capture.stop()

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🎯 AI Crowd Monitoring System - Backend")
    print("=" * 60)
    print(f"YOLO Available: {YOLO_AVAILABLE}")
    print(f"Face Recognition Available: {FACE_RECOGNITION_AVAILABLE}")
    print("\nStarting server on http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, threaded=True, port=5000)
