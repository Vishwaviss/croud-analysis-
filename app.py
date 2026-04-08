

import cv2
import numpy as np
import os
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import threading
import time
from collections import deque
from datetime import datetime
import base64
from io import BytesIO


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


app = Flask(__name__, template_folder=".", static_folder=".", static_url_path="")
CORS(app)

CONFIG = {
    'VIDEO_UPLOAD_FOLDER': 'uploads/',
    'MAX_FRAME_BUFFER': 30,
    'DETECTION_THRESHOLD': 0.3,
    'HEAT_MAP_GRID_SIZE': 40,
    'MAX_FACES_STORED': 50,
}


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
        self.known_face_encodings = []
        self.known_face_names = []
        self.seen_session_ids = set() # Store all unique track IDs seen in this session
        self.active_tracks = {} # Store current frame track IDs to name mapping
        self.track_id_to_name = {} # Persistent Mapping: track_id -> name (recognized or assigned)
        self.track_start_times = {} # When each track was first seen
        self.best_face_crops = {} # Store best face base64 crop for each track_id 
        self.face_processing_queue = deque(maxlen=20)
        self.is_face_processing = False

system_state = SystemState()


def load_models():
    """Load AI models with hardware acceleration"""
    models = {}
    
    # DETERMINE DEVICE
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if YOLO_AVAILABLE:
        try:
            # Load nano model for speed
            model = YOLO('yolov8n.pt')
            
            # AUTOMATIC HARDWARE ACCELERATION:
            if device == 'cuda':
                model.to(device)
                model.half() # Use FP16 for double speed on GPU
            
            print(f"✅ YOLO Mega Stealth loaded on: {device.upper()}")
            models['yolo'] = model
        except Exception as e:
            print(f"❌ Failed to load YOLO: {e}")
    
    return models

models = load_models()

def detect_people(frame, model, threshold=0.5):
    """
    Detect AND Track people using YOLO track
    Returns: count, detections list, annotated frame
    """
    if not YOLO_AVAILABLE or 'yolo' not in models:
        return 0, [], frame
    
    try:
        # Use ultra-lean imgsz for HYPER-SPEED mode
        results = model.track(frame, imgsz=320, conf=threshold, persist=True, verbose=False)
        
        detections = []
        people_count = 0
        
        if results and len(results) > 0 and results[0].boxes:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                if cls_id != 0: continue # Only 'person' class
                
                conf_val = float(box.conf.item())
                
                # Get Track ID if available, else use a placeholder
                track_id = int(box.id.item()) if box.id is not None else None
                
                people_count += 1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf_val,
                    'track_id': track_id,
                    'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                })
        
        return people_count, detections, frame
    
    except Exception as e:
        print(f"❌ AI Error: {e}")
        return 0, [], frame
    
    except Exception as e:
        print(f"❌ AI Error: {e}")
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
        
        # Find faces ONLY inside the detected person bounding boxes
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.get('bbox', (0,0,0,0))
            
            # Estimate upper body (top 60%) for better face detection
            h = y2 - y1
            y2_upper = min(y2, int(y1 + h * 0.6))
            
            # Ensure coordinates are valid and within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2_upper = min(frame.shape[1], x2), min(frame.shape[0], y2_upper)
            
            if y2_upper <= y1 or x2 <= x1:
                continue
                
            crop_rgb = rgb_frame[y1:y2_upper, x1:x2]
            # Ensure crop is contiguous in memory to avoid dlib TypeError
            crop_rgb = np.ascontiguousarray(crop_rgb)
            
            # Run face detection on the small crop
            face_locs = face_recognition.face_locations(crop_rgb, model='hog')
            if not face_locs:
                continue
                
            # Get encodings for faces in this crop
            encodings = face_recognition.face_encodings(crop_rgb, face_locs)
            
            for (crop_top, crop_right, crop_bottom, crop_left), encoding in zip(face_locs, encodings):
                # Map coordinates back to the full frame
                top = y1 + crop_top
                bottom = y1 + crop_bottom
                left = x1 + crop_left
                right = x1 + crop_right
                
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

# ============================================================================
# Attendance and ID Recognition Logic
# ============================================================================
import csv

def log_attendance(name):
    """Log person arrival time to CSV file"""
    file_exists = os.path.isfile('attendance_log.csv')
    try:
        with open('attendance_log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Name', 'Date', 'Time'])
            
            now = datetime.now()
            writer.writerow([name, now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')])
    except Exception as e:
        print(f"Logging error: {e}")

def load_face_database():
    """Load pre-registered people from known_faces directory"""
    if not os.path.exists('known_faces'):
        os.makedirs('known_faces')
        return
        
    print("📡 Loading pre-registered identities...")
    for filename in os.listdir('known_faces'):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            name = os.path.splitext(filename)[0]
            try:
                img = face_recognition.load_image_file(f"known_faces/{filename}")
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    system_state.known_face_encodings.append(encodings[0])
                    system_state.known_face_names.append(name)
                    print(f"✅ Integrated Registered Identity: {name}")
            except Exception as e:
                print(f"⚠️ Error loading {filename}: {e}")

def recognize_faces(frame, faces, can_assign_new=True):
    """Identify faces, handle registration and logging"""
    if not os.path.exists('captured_faces'):
        os.makedirs('captured_faces')

    if not faces or not FACE_RECOGNITION_AVAILABLE:
        return []
    
    recognized = []
    try:
        for face in faces:
            encoding = face['encoding']
            top, right, bottom, left = face['location']
            
            # ONLY compare against valid encodings
            valid_indices = [i for i, enc in enumerate(system_state.known_face_encodings) if enc is not None]
            valid_encodings = [system_state.known_face_encodings[i] for i in valid_indices]
            
            name = None
            found_match = False
            
            if valid_encodings:
                try:
                    matches = face_recognition.compare_faces(valid_encodings, encoding, tolerance=0.55)
                    if True in matches:
                        first_match_index = matches.index(True)
                        actual_index = valid_indices[first_match_index]
                        name = system_state.known_face_names[actual_index]
                        found_match = True
                except Exception as e:
                    print(f"⚠️ Comparison error: {e}")

            if not found_match and can_assign_new:
                # ONLY create new ID once the scan period completes 
                system_state.total_count += 1
                name = f"Identity_{system_state.total_count}"
                print(f"✨ Identity Assigned: {name}")
                system_state.known_face_encodings.append(encoding)
                system_state.known_face_names.append(name)
                
                # Save sighting record
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                face_filename = f"captured_faces/{name}_{timestamp}.jpg"
                cv2.imwrite(face_filename, face['image'])
                log_attendance(name)
                
                add_alert(f"👤 Person Identified: {name}", 'info')
                found_match = True

            if name: # If we found a match OR created a new one
                recognized.append({
                    'id': name,
                    'location': (left, top, right, bottom),
                    'confidence': face['confidence'],
                    'timestamp': time.time()
                })
                
                # Prepare base64 for UI preview
                ret, buffer = cv2.imencode('.jpg', face['image'])
                image_base64 = ""
                if ret:
                    image_base64 = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode()
                
                system_state.best_face_crops[name] = image_base64

                display_face = {
                    'id': name,
                    'location': (left, top, right, bottom),
                    'confidence': face['confidence'],
                    'timestamp': time.time(),
                    'image_base64': image_base64
                }
                system_state.detected_faces.append(display_face)
            
    except Exception as e:
        print(f"Face recognition error: {e}")
    
    return recognized

# ============================================================================
# Data Processing Functions
# ============================================================================

import pickle

def save_encodings():
    """Save known face encodings to a file"""
    try:
        data = {
            'encodings': system_state.known_face_encodings,
            'names': system_state.known_face_names
        }
        with open('encodings.pkl', 'wb') as f:
            pickle.dump(data, f)
        # print("✅ Face encodings saved")
    except Exception as e:
        print(f"Error saving encodings: {e}")

def load_encodings():
    """Load known face encodings from a file"""
    try:
        if os.path.exists('encodings.pkl'):
            with open('encodings.pkl', 'rb') as f:
                data = pickle.load(f)
                raw_encodings = data.get('encodings', [])
                raw_names = data.get('names', [])
                
                # Cleanup: Filter out None values to prevent broadcasting errors
                system_state.known_face_encodings = []
                system_state.known_face_names = []
                for enc, name in zip(raw_encodings, raw_names):
                    if enc is not None:
                        system_state.known_face_encodings.append(enc)
                        system_state.known_face_names.append(name)
                
                system_state.total_count = len(system_state.known_face_names)
            print(f"✅ Loaded {len(system_state.known_face_names)} valid identities")
    except Exception as e:
        print(f"Error loading encodings: {e}")

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

def face_recognition_worker():
    """Background worker to handle face recognition without blocking the main video loop"""
    while True:
        if system_state.face_processing_queue:
            task = system_state.face_processing_queue.popleft()
            frame = task['frame']
            detections = task['detections']
            # track_id mapping is already in SystemState, we update the name when found
            can_assign_new = task['detections'][0].get('can_assign_new', False)
            faces = extract_faces(frame, detections)
            if faces:
                recognized = recognize_faces(frame, faces, can_assign_new=can_assign_new)
                if recognized:
                    for r in recognized:
                        # Find the track ID this face belongs to
                        # In this simple implementation, we just update the track for which we ran face recognition
                        track_id = detections[0].get('track_id')
                        if track_id:
                            system_state.track_id_to_name[track_id] = r['id']
            # Small yield after processing to prevent CPU hogging
            time.sleep(0.01)
        else:
            time.sleep(0.1) # Longer sleep when idle

# Start face recognition worker thread
threading.Thread(target=face_recognition_worker, daemon=True).start()

class VideoCapture:
    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self.lock = threading.Lock()
        self.is_running = False
    
    def start(self):
        """Start video capture with robust error handling and fallbacks"""
        print(f"📸 Attempting to open video source: {self.source}")
        
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, None] # Try multiple backends on Windows
        indices = [self.source, 0, 1] if isinstance(self.source, int) else [self.source]
        
        opened = False
        for idx in indices:
            for backend in backends:
                try:
                    if backend is not None:
                        self.cap = cv2.VideoCapture(idx, backend)
                    else:
                        self.cap = cv2.VideoCapture(idx)
                    
                    if self.cap.isOpened():
                        # Try to read a test frame
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            print(f"✅ Successfully opened camera {idx} with backend {backend}")
                            opened = True
                            break
                        else:
                            self.cap.release()
                except Exception as e:
                    print(f"⚠️ Failed camera {idx} with backend {backend}: {e}")
                    continue
            if opened: break
            
        if not opened:
            print("❌ ALL CAMERA FALLBACKS FAILED.")
            # Don't raise, instead allow the system to simulate or wait
            return False
            
        # Optimization: Don't buffer old frames
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        self.raw_frame = None
        self.last_detections = []
        self.stopped = False
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.capture_thread.start()
        
        # Start AI inference thread (Asynchronous)
        self.ai_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.ai_thread.start()
        
        # Start processing thread (Visualization)
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
    
    def _reader_loop(self):
        """MAX SPEED: Dedicated thread to grab raw frames from hardware"""
        while self.is_running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        self.raw_frame = frame
                    else:
                        time.sleep(0.01)
                else:
                    time.sleep(1.0)
            except Exception as e:
                print(f"⚠️ Capture Thread Error: {e}")
                time.sleep(0.5)

    def _inference_loop(self):
        """Asynchronous AI Inference - Runs as fast as possible in its own thread"""
        frame_count = 0
        while self.is_running:
            try:
                if self.raw_frame is None or self.raw_frame.size == 0:
                    time.sleep(0.1)
                    continue
                    
                # Grab latest frame for AI
                frame_for_ai = self.raw_frame.copy()
                frame_count += 1
                
                # RUN AI - Using smaller internal size for MAX SPEED
                people_count, detections, _ = detect_people(
                    frame_for_ai, models.get('yolo'), system_state.threshold
                )
                
                self.last_detections = detections
                system_state.people_count = people_count
                
                # Process recognition and metadata logic asynchronously
                for det in detections:
                    track_id = det.get('track_id')
                    if track_id is not None:
                        current_name = system_state.track_id_to_name.get(track_id, "")
                        
                        # 4-SECOND INTELLIGENT SCAN LOGIC
                        if track_id not in system_state.track_start_times:
                            system_state.track_start_times[track_id] = time.time()
                            system_state.track_id_to_name[track_id] = "Scanning..."
                            system_state.seen_session_ids.add(track_id)

                        time_elapsed = time.time() - system_state.track_start_times[track_id]
                        
                        # PRIORITY: Assign identity if still scanning and window closes
                        if current_name == "Scanning..." and time_elapsed > 5.0:
                            system_state.total_count += 1
                            new_name = f"Identity_{system_state.total_count}"
                            system_state.track_id_to_name[track_id] = new_name
                            
                            # Log to attendance
                            log_attendance(new_name)
                            # We don't add to known_face_encodings yet because we lack one.
                            # The person will be tracked in this session, and if a face is found later, 
                            # recognize_faces will update the encoding.
                            
                            # Permanent Disk Storage (Fail-Safe)
                            if not os.path.exists('captured_faces'):
                                os.makedirs('captured_faces')
                                
                            best_b64 = system_state.best_face_crops.get(track_id, "")
                            # Also save a timestamped JPG for persistent history
                            try:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filepath = f"captured_faces/{new_name}_{timestamp}.jpg"
                                # Convert base64 back to image for saving if needed, 
                                # but we'll try to save the raw crop if available from worker loop
                                # For simplicity, we just use the last best_face_crop updated in loop
                                if track_id in system_state.best_face_crops:
                                    # Since best_face_crops was just updated from frame_ai
                                    pass
                            except: pass
                            
                            system_state.detected_faces.append({
                                'id': new_name,
                                'confidence': 0.8,
                                'location': (0,0,0,0),
                                'timestamp': time.time(),
                                'image_base64': best_b64
                            })
                            
                            add_alert(f"👤 Identity Assigned: {new_name}", 'info')
                            # Log to attendance
                            log_attendance(new_name)
                            should_reprocess = False 
                        else:
                            should_reprocess = "Scanning..." in current_name or "Identity_" in current_name
                            can_assign_new = time_elapsed > 4.0
                        
                        # Increased scanning frequency (every 5 frames)
                        if should_reprocess and (frame_count % 5 == 0):
                            x1, y1, x2, y2 = det['bbox']
                            h, w = frame_for_ai.shape[:2]
                            margin = 40
                            y1_m, y2_m = max(0, y1-margin), min(h, y2+margin)
                            x1_m, x2_m = max(0, x1-margin), min(w, x2+margin)
                            
                            face_crop = frame_for_ai[y1_m:y2_m, x1_m:x2_m]
                            if face_crop.size > 0:
                                # Update best known thumbnail for both UI and DISK
                                try:
                                    ret, buffer = cv2.imencode('.jpg', face_crop)
                                    if ret:
                                        system_state.best_face_crops[track_id] = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode()
                                        
                                        # If after 5s we forced an identity, save this best face to DISK
                                        if current_name != "Scanning..." and "Identity_" in current_name:
                                            # Ensure we save at least one photo for this ID if not already saved
                                            # Use a flag or check if filename exists
                                            pass
                                except: pass

                                system_state.face_processing_queue.appendleft({
                                    'frame': face_crop.copy(),
                                    'detections': [{'bbox': (0, 0, face_crop.shape[1], face_crop.shape[0]), 
                                                   'track_id': track_id, 
                                                   'can_assign_new': can_assign_new}]
                                })

                if detections:
                    system_state.heatmap_data = generate_heatmap_data(detections, frame_for_ai.shape)
                    system_state.scatter_points.extend(generate_scatter_points(detections))
                    system_state.anomalies = detect_anomalies(detections, frame_for_ai.shape)

                time.sleep(0.01)
            except Exception as e:
                print(f"⚠️ AI Inference Error: {e}")
                time.sleep(0.1)
    
    def stop(self):
        """Stop video capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
    
    def _process_loop(self):
        """UI Drawing and Visualization thread - decoupled from AI and Capture"""
        DISPLAY_W, DISPLAY_H = 1280, 720
        while self.is_running:
            try:
                if self.raw_frame is None or self.raw_frame.size == 0:
                    time.sleep(0.1)
                    continue
                    
                # 1. Resize for display (Speed: NEAREST)
                frame_display = cv2.resize(self.raw_frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_NEAREST)
                
                # 2. Get scaling factors
                raw_h, raw_w = self.raw_frame.shape[:2]
                if raw_w == 0 or raw_h == 0: continue
                sx, sy = DISPLAY_W / raw_w, DISPLAY_H / raw_h
                
                detections = self.last_detections
                
                # 3. Render detections with coordinate scaling
                for det in detections:
                    rx1, ry1, rx2, ry2 = det['bbox']
                    x1, y1 = int(rx1 * sx), int(ry1 * sy)
                    x2, y2 = int(rx2 * sx), int(ry2 * sy)
                    
                    tid = det.get('track_id')
                    name = system_state.track_id_to_name.get(tid, f"ID:{tid}")
                    
                    color = (0, 255, 127) # PRO GREEN
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
                    
                    label_size, _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
                    label_y = max(y1, 30)
                    cv2.rectangle(frame_display, (x1, label_y - 30), (x1 + label_size[0] + 10, label_y), color, -1)
                    cv2.putText(frame_display, name, (x1 + 5, label_y - 8), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)

                with self.lock:
                    system_state.current_frame = frame_display
                    current_time = time.time()
                    system_state.frame_times.append(current_time)
                    if len(system_state.frame_times) > 1:
                        system_state.fps = len(system_state.frame_times) / (system_state.frame_times[-1] - system_state.frame_times[0])
                
                time.sleep(0.001)
            except Exception as e:
                print(f"⚠️ Process Thread Error: {e}")
                time.sleep(0.1)

video_capture = VideoCapture(0)  # Default to webcam

# ============================================================================
# Crowd Analysis Helpers
# ============================================================================

def generate_heatmap_data(detections, frame_shape):
    """
    Generate a 10x10 density grid for the frontend heatmap.
    Returns: list of 100 values [0.0 to 1.0]
    """
    grid_size = 10
    grid = [0.0] * (grid_size * grid_size)
    
    if not detections or frame_shape[0] == 0 or frame_shape[1] == 0:
        return grid
        
    h, w = frame_shape[:2]
    
    for det in detections:
        # Use center of bounding box
        x1, y1, x2, y2 = det['bbox']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Map to grid
        col = int((cx / w) * grid_size)
        row = int((cy / h) * grid_size)
        
        # Clamp values
        col = max(0, min(grid_size - 1, col))
        row = max(0, min(grid_size - 1, row))
        
        grid[row * grid_size + col] += 0.2 # Each person adds density
        
    # Standardize to 0.0 - 1.0 range
    return [min(1.0, val) for val in grid]

def generate_scatter_points(detections):
    """
    Generate simplified (x, y) coordinates for the scatter plot.
    Returns: list of {'x': float, 'y': float}
    """
    points = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Approximate relative position (0.0 to 1.0)
        # Using fixed 720p/HD assumptions since script.js expects a chart space
        points.append({
            'x': round(cx, 1),
            'y': round(cy, 1)
        })
    return points

def detect_anomalies(detections, frame_shape):
    """
    Detect unusual patterns (e.g. high density clustering).
    """
    anomalies = 0
    count = len(detections)
    
    # Simple overcrowding threshold
    if count > 20: 
        anomalies += 1
        
    # High density cluster detection
    if count > 5:
        centers = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
            
        for i, (cx1, cy1) in enumerate(centers):
            nearby = 0
            for j, (cx2, cy2) in enumerate(centers):
                if i == j: continue
                # If people are within 80 pixels of each other
                if abs(cx1 - cx2) < 80 and abs(cy1 - cy2) < 80:
                    nearby += 1
            
            if nearby > 4: # Cluster found
                anomalies += 1
                break
                
    return min(anomalies, 5)

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

def jls_extract_def():
    return 'index.html'


@app.route('/')
def index():
    """Serve main HTML"""
    return render_template(jls_extract_def())

@app.route('/video_feed')
def video_feed():
    """Stream video frames"""
    def generate():
        while True:
            frame = system_state.current_frame
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Encode frame with HIGH QUALITY (95%)
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
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
    
    # Prepare faces data (already pre-encoded in system_state)
    faces_data = list(system_state.detected_faces)
    
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
    
    elif action == 'reset_all':
        try:
            # 1. Reset memory state
            system_state.total_count = 0
            system_state.anomalies = 0
            system_state.alerts.clear()
            system_state.known_face_encodings = []
            system_state.known_face_names = []
            system_state.track_id_to_name = {}
            system_state.seen_session_ids.clear()
            system_state.detected_faces.clear()
            
            # 2. Delete persistence files
            if os.path.exists('encodings.pkl'):
                os.remove('encodings.pkl')
            
            # 3. Clear captured folders
            import shutil
            for folder in ['captured_faces', 'captured_persons']:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                    os.makedirs(folder)
            
            add_alert("🚨 System completely reset!", 'danger')
            print("🚨 SYSTEM RESET BY USER")
            return jsonify({'status': 'full_reset_ok'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
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

import atexit
startup_done = False

@app.before_request
def startup():
    """Initialize on first request with student database loading"""
    global startup_done
    if not startup_done and not video_capture.is_running:
        try:
            # 1. Load pre-registered students/staff from 'known_faces'
            load_face_database()
            
            # 2. Load session-to-session face data
            load_encodings()
            
            video_capture.start()
            print("✅ Smart Attendance Module Activated")
            atexit.register(video_capture.stop)
            atexit.register(save_encodings)
        except Exception as e:
            print(f"❌ Module Initialization Error: {e}")
        startup_done = True

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
    
    app.run(debug=True, threaded=True, port=5000, use_reloader=False)
