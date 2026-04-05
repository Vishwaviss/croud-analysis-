"""
Advanced Examples & Extensions for Crowd Monitoring System
Add these features to enhance your system
"""

# ============================================================================
# 1. DATABASE INTEGRATION (SQLAlchemy + SQLite)
# ============================================================================

"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class DetectionRecord(Base):
    __tablename__ = 'detections'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    people_count = Column(Integer)
    fps = Column(Float)
    anomalies = Column(Integer)
    heatmap_data = Column(String)  # JSON string
    
    def __repr__(self):
        return f"Detection({self.timestamp}, people={self.people_count})"

class FaceRecord(Base):
    __tablename__ = 'faces'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    face_encoding = Column(String)  # Base64 encoded
    confidence = Column(Float)
    face_id = Column(String)
    
# Initialize database
engine = create_engine('sqlite:///detections.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def save_detection(people_count, fps, anomalies, heatmap):
    session = Session()
    record = DetectionRecord(
        people_count=people_count,
        fps=fps,
        anomalies=anomalies,
        heatmap_data=str(heatmap)
    )
    session.add(record)
    session.commit()
    session.close()
"""

# ============================================================================
# 2. EMAIL ALERTS
# ============================================================================

"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

class EmailAlertService:
    def __init__(self, sender_email, sender_password, smtp_server='smtp.gmail.com', smtp_port=587):
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
    
    def send_alert(self, recipient_email, subject, message, priority='normal'):
        try:
            # Create email
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = f"[{priority.upper()}] {subject}"
            
            # Email body
            body = f"""
            Alert: {subject}
            Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Message: {message}
            
            ---
            Crowd Monitoring System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            print(f"✅ Email sent to {recipient_email}")
            return True
        
        except Exception as e:
            print(f"❌ Failed to send email: {e}")
            return False

# Usage in app.py:
# alert_service = EmailAlertService('your_email@gmail.com', 'your_password')
# alert_service.send_alert('admin@example.com', 'High Crowd Detected', 
#                          f'People count: 150', priority='high')
"""

# ============================================================================
# 3. WEBHOOK ALERTS
# ============================================================================

"""
import requests
import json
from datetime import datetime

class WebhookAlertService:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
    
    def send_alert(self, alert_type, data):
        payload = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': alert_type,
            'data': data,
            'system': 'crowd_monitoring',
        }
        
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Webhook error: {e}")
            return False

# Usage:
# webhook = WebhookAlertService('https://example.com/webhook')
# webhook.send_alert('high_crowd', {'count': 150, 'location': 'Zone A'})
"""

# ============================================================================
# 4. CUSTOM ANOMALY DETECTION
# ============================================================================

"""
from collections import deque
import numpy as np

class AnomalyDetector:
    def __init__(self, window_size=10, threshold=2.0):
        self.window_size = window_size
        self.threshold = threshold
        self.history = deque(maxlen=window_size)
    
    def detect(self, current_count):
        self.history.append(current_count)
        
        if len(self.history) < self.window_size:
            return False, None
        
        # Calculate statistics
        mean = np.mean(self.history)
        std = np.std(self.history)
        
        # Detect sudden spikes
        z_score = abs((current_count - mean) / (std + 1e-5))
        
        if z_score > self.threshold:
            return True, {
                'type': 'sudden_change',
                'z_score': z_score,
                'mean': mean,
                'current': current_count
            }
        
        return False, None

# Usage:
# detector = AnomalyDetector()
# is_anomaly, details = detector.detect(people_count)
"""

# ============================================================================
# 5. FACE RECOGNITION WITH KNOWN FACES DATABASE
# ============================================================================

"""
import face_recognition
import pickle
import os
from pathlib import Path

class FaceDatabase:
    def __init__(self, database_path='known_faces/'):
        self.database_path = database_path
        self.known_faces = {}
        self.load_database()
    
    def load_database(self):
        # Load known faces from disk
        db_file = os.path.join(self.database_path, 'faces.pkl')
        
        if os.path.exists(db_file):
            with open(db_file, 'rb') as f:
                self.known_faces = pickle.load(f)
            print(f"✅ Loaded {len(self.known_faces)} known faces")
        else:
            print("No known faces database found")
    
    def add_face(self, person_name, image_path):
        # Load image and encode face
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            if person_name not in self.known_faces:
                self.known_faces[person_name] = []
            self.known_faces[person_name].append(encodings[0])
            self.save_database()
            print(f"✅ Added {person_name}")
        else:
            print(f"❌ No face found in {image_path}")
    
    def save_database(self):
        db_file = os.path.join(self.database_path, 'faces.pkl')
        os.makedirs(self.database_path, exist_ok=True)
        
        with open(db_file, 'wb') as f:
            pickle.dump(self.known_faces, f)
    
    def recognize_face(self, face_encoding, tolerance=0.6):
        # Compare with known faces
        for person_name, encodings in self.known_faces.items():
            matches = face_recognition.compare_faces(
                encodings, face_encoding, tolerance=tolerance
            )
            if any(matches):
                return person_name
        return None

# Usage:
# db = FaceDatabase()
# db.add_face('suspect_1', 'path/to/image.jpg')
# recognized = db.recognize_face(face_encoding)
"""

# ============================================================================
# 6. VIDEO RECORDING
# ============================================================================

"""
import cv2
import os
from datetime import datetime

class VideoRecorder:
    def __init__(self, output_dir='recordings/', fps=30, resolution=(640, 480)):
        self.output_dir = output_dir
        self.fps = fps
        self.resolution = resolution
        self.writer = None
        self.is_recording = False
        os.makedirs(output_dir, exist_ok=True)
    
    def start(self):
        # Create video writer
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.output_dir, f'recording_{timestamp}.mp4')
        
        # Use H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(filename, fourcc, self.fps, self.resolution)
        
        self.is_recording = True
        print(f"✅ Recording started: {filename}")
    
    def write_frame(self, frame):
        if self.is_recording and self.writer:
            # Ensure frame matches resolution
            frame = cv2.resize(frame, self.resolution)
            self.writer.write(frame)
    
    def stop(self):
        if self.writer:
            self.writer.release()
        self.is_recording = False
        print("✅ Recording stopped")

# Usage:
# recorder = VideoRecorder()
# recorder.start()
# recorder.write_frame(frame)
# recorder.stop()
"""

# ============================================================================
# 7. ADVANCED CROWD ANALYSIS
# ============================================================================

"""
import numpy as np
from scipy.ndimage import label

class CrowdAnalyzer:
    def __init__(self, grid_size=40):
        self.grid_size = grid_size
    
    def analyze_distribution(self, detections, frame_shape):
        h, w = frame_shape[:2]
        
        # Create density grid
        grid_h = h // self.grid_size
        grid_w = w // self.grid_size
        density_grid = np.zeros((grid_h, grid_w))
        
        for det in detections:
            cx, cy = det['center']
            gx, gy = min(cx // self.grid_size, grid_w - 1), min(cy // self.grid_size, grid_h - 1)
            density_grid[gy, gx] += 1
        
        # Find clusters
        labeled_array, num_features = label(density_grid > 0)
        
        clusters = []
        for i in range(1, num_features + 1):
            cluster_positions = np.where(labeled_array == i)
            cluster_size = len(cluster_positions[0])
            
            if cluster_size > 3:  # Only significant clusters
                cy = np.mean(cluster_positions[0]) * self.grid_size
                cx = np.mean(cluster_positions[1]) * self.grid_size
                
                clusters.append({
                    'center': (int(cx), int(cy)),
                    'size': cluster_size,
                    'density': cluster_size / (cluster_size * self.grid_size ** 2)
                })
        
        return {
            'total_clusters': len(clusters),
            'clusters': clusters,
            'density_grid': density_grid,
            'max_density': np.max(density_grid),
            'avg_density': np.mean(density_grid),
        }

# Usage:
# analyzer = CrowdAnalyzer()
# results = analyzer.analyze_distribution(detections, frame.shape)
"""

# ============================================================================
# 8. PERFORMANCE MONITORING
# ============================================================================

"""
import time
from collections import deque
import psutil

class PerformanceMonitor:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.inference_times = deque(maxlen=window_size)
        self.start_time = time.time()
    
    def record_frame_time(self, elapsed):
        self.frame_times.append(elapsed)
    
    def record_inference_time(self, elapsed):
        self.inference_times.append(elapsed)
    
    def get_fps(self):
        if len(self.frame_times) < 2:
            return 0
        total_time = self.frame_times[-1] - self.frame_times[0]
        return len(self.frame_times) / max(total_time, 0.001)
    
    def get_stats(self):
        return {
            'fps': self.get_fps(),
            'avg_frame_time': np.mean(self.frame_times) if self.frame_times else 0,
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0,
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'uptime': time.time() - self.start_time,
        }

# Usage:
# monitor = PerformanceMonitor()
# while True:
#     start = time.time()
#     # process frame
#     monitor.record_frame_time(start)
#     stats = monitor.get_stats()
"""

# ============================================================================
# 9. CUSTOM FLASK ROUTES EXAMPLE
# ============================================================================

"""
from flask import request, send_file
import io

# In app.py, add these routes:

@app.route('/api/download-report')
def download_report():
    # Generate PDF or CSV report
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    c.drawString(100, 750, f"Crowd Monitoring Report")
    c.drawString(100, 700, f"Total People Detected: {system_state.total_count}")
    c.drawString(100, 650, f"Peak Count: {system_state.people_count}")
    c.drawString(100, 600, f"Anomalies: {system_state.anomalies}")
    
    c.save()
    buffer.seek(0)
    
    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='report.pdf'
    )

@app.route('/api/export-data')
def export_data():
    import csv
    
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    
    writer.writerow(['Timestamp', 'People Count', 'Anomalies', 'FPS'])
    # Write historical data here
    
    return {
        'data': buffer.getvalue(),
        'filename': 'detections.csv'
    }
"""

# ============================================================================
# 10. INTEGRATION WITH EXTERNAL SYSTEMS
# ============================================================================

"""
# Integration with security system
def trigger_security_alert(alert_level, people_count):
    if alert_level == 'high':
        # Call security API
        requests.post('https://security-api.example.com/alert', json={
            'level': 'high',
            'people_count': people_count,
            'location': 'main_entrance',
            'action': 'notify_security'
        })
    
    elif alert_level == 'critical':
        # Lock doors, start recording
        requests.post('https://security-api.example.com/lockdown', json={
            'zones': ['entrance', 'hallway'],
            'start_recording': True,
        })

# Integration with camera system
def get_camera_feeds():
    cameras = [
        {'id': 1, 'url': 'rtsp://192.168.1.100:554/stream1'},
        {'id': 2, 'url': 'rtsp://192.168.1.101:554/stream1'},
        {'id': 3, 'url': 'rtsp://192.168.1.102:554/stream1'},
    ]
    return cameras

# Integration with crowd management app
def notify_crowd_management_app(data):
    requests.post('https://app.example.com/api/crowd-update', json={
        'zone': data['zone'],
        'capacity': data['capacity'],
        'current_count': data['current_count'],
        'percentage': (data['current_count'] / data['capacity']) * 100,
        'recommendation': 'increase entry' if data['current_count'] < data['capacity'] else 'halt entry',
    })
"""

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

"""
1. Copy desired code snippet to your app.py or separate file
2. Import required libraries
3. Initialize class/function in app initialization
4. Call functions as needed

Example:
    from config import AI_CONFIG
    from advanced_extensions import EmailAlertService
    
    # In main app
    alert_service = EmailAlertService('your_email@gmail.com', 'password')
    
    # When anomaly detected
    if anomalies > threshold:
        alert_service.send_alert(
            'admin@example.com',
            'High Crowd Detected',
            f'People count: {people_count}',
            priority='high'
        )
"""

print(__doc__)
