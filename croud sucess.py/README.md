# 🎯 AI Crowd Monitoring System

A real-time **AI-powered crowd monitoring system** with people detection, face recognition, heatmap visualization, and anomaly detection.

## 📋 Features

### Core Features
- ✅ **Real-time People Detection** - Uses YOLOv8 for accurate person detection
- ✅ **Live Video Feed** - Stream from webcam or uploaded video files
- ✅ **Crowd Density Heatmap** - Visualize density distribution across the frame
- ✅ **Scatter Plot Analysis** - Show person positions in real-time
- ✅ **Face Recognition** - Detect and identify faces in crowds
- ✅ **Anomaly Detection** - Alert when unusual crowd patterns detected
- ✅ **FPS Monitoring** - Track processing speed
- ✅ **Responsive Dashboard** - Modern web-based UI with real-time updates

### AI/ML Models Used
1. **YOLOv8 (Nano)** - Person detection and counting
2. **Face Recognition** - Face detection and recognition
3. **OpenCV** - Image processing and visualization

### Visualizations
- 📊 Live video feed with bounding boxes
- 🔥 Crowd density heatmap
- 📍 Scatter chart of detected persons
- 🎭 Detected faces gallery
- ⚠️ Real-time alerts panel

---

## 🚀 Quick Start

### 1. **Installation**

#### Clone/Setup Project
```bash
# Create project directory
mkdir crowd-monitoring
cd crowd-monitoring

# Copy files (index.html, style.css, script.js, app.py)
# into this directory
```

#### Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Note:** Some packages may require additional setup:
- **dlib**: May need C++ compiler (Visual Studio Build Tools on Windows)
- **face_recognition**: Depends on dlib

#### Install dlib (Alternative)
```bash
# Windows (if pip install fails)
pip install cmake
pip install dlib

# macOS
brew install cmake
pip install dlib

# Linux
sudo apt-get install cmake
pip install dlib
```

### 2. **Project Structure**

```
crowd-monitoring/
├── app.py                 # Flask backend
├── index.html             # Frontend HTML
├── style.css              # Frontend styles
├── script.js              # Frontend logic
├── requirements.txt       # Python dependencies
├── uploads/               # Video upload folder
└── README.md             # This file
```

### 3. **Create Templates Folder**

```bash
mkdir -p templates
mv index.html templates/
```

Your folder structure should be:
```
crowd-monitoring/
├── app.py
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── uploads/
├── requirements.txt
└── README.md
```

### 4. **Update app.py Routes**

Update the route in `app.py`:
```python
@app.route('/')
def index():
    return render_template('index.html')
```

And create `app.py` with static file serving:
```python
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)
```

### 5. **Run the System**

```bash
# Start Flask server
python app.py

# Open in browser
# http://localhost:5000
```

---

## 🎮 Usage Guide

### Dashboard Overview

#### **Top Section**
- Connection status indicator
- Current time display
- Live FPS counter

#### **Left Panel**
- **Live Feed**: Real-time video with person count overlay
- **Controls**:
  - Choose video source (Webcam/Upload)
  - Adjust detection threshold
  - View live statistics

#### **Middle Panel**
- **Heatmap**: Color-coded crowd density visualization
- **Statistics**: Total count, density level, FPS, anomalies

#### **Right Panel**
- **Scatter Plot**: Detection positions in real-time
- **Suspicious Faces**: Gallery of detected faces with confidence scores

#### **Bottom Section**
- **Alerts Panel**: Real-time system alerts and anomalies

### Key Controls

| Control | Action |
|---------|--------|
| Radio buttons | Switch between webcam and video upload |
| Upload button | Choose a video file to analyze |
| Threshold slider | Adjust detection sensitivity (0.1-0.9) |

### Real-Time Metrics

- **People Count**: Current frame person count
- **Total Count**: Cumulative count session
- **Density Level**: Low/Medium/High based on count
- **Anomalies**: Number of detected anomalies
- **FPS**: Processing speed in frames per second

---

## 🔧 Configuration

Edit settings in `app.py`:

```python
CONFIG = {
    'VIDEO_UPLOAD_FOLDER': 'uploads/',      # Video storage
    'MAX_FRAME_BUFFER': 30,                 # Frame buffer size
    'DETECTION_THRESHOLD': 0.5,             # Detection confidence
    'HEAT_MAP_GRID_SIZE': 40,              # Heatmap grid cell size
    'MAX_FACES_STORED': 50,                # Max faces to store
}
```

### Adjust Detection Threshold
- **Lower (0.1-0.3)**: More detections, more false positives
- **Medium (0.5)**: Balanced (default)
- **Higher (0.7-0.9)**: Fewer detections, more accurate

---

## 📊 API Endpoints

### GET /api/detection-data
Returns real-time detection data:
```json
{
  "people_count": 15,
  "total_count": 150,
  "fps": 24.5,
  "anomalies": 2,
  "heatmap_data": [...],
  "scatter_points": [...],
  "detected_faces": [...],
  "new_alerts": [...]
}
```

### POST /api/command
Send commands to system:
```json
{
  "action": "update_threshold",
  "value": 0.6
}
```

### POST /api/upload-video
Upload video file for processing

### GET /api/status
Get system status and model availability

---

## 🧠 AI Models Explained

### YOLOv8 (You Only Look Once)
- **Purpose**: Real-time object detection
- **Speed**: ~24-30 FPS on CPU, 60+ on GPU
- **Accuracy**: Detects persons in crowd
- **Config**: Using "nano" (smallest, fastest)

**To use larger models:**
```python
# In app.py, change:
models['yolo'] = YOLO('yolov8s.pt')  # Small
models['yolo'] = YOLO('yolov8m.pt')  # Medium (better accuracy)
models['yolo'] = YOLO('yolov8l.pt')  # Large (slowest)
```

### Face Recognition
- **Purpose**: Detect and identify faces
- **Method**: Deep learning-based face encoding
- **Use**: Identify suspicious persons in crowd
- **Config**: HOG (CPU) or CNN (GPU) detection

### Heatmap Generation
- **Purpose**: Visualize crowd density distribution
- **Method**: Grid-based density calculation
- **Color**: Blue (low) → Red (high density)

---

## 📈 Performance Tips

### Improve Speed
```python
# 1. Reduce frame resolution
frame = cv2.resize(frame, (320, 240))  # Instead of 640x480

# 2. Use GPU acceleration
# Set CUDA_VISIBLE_DEVICES environment variable

# 3. Reduce detection frequency
# Process every 2nd frame: frame_count % 2 == 0

# 4. Smaller YOLO model
models['yolo'] = YOLO('yolov8n.pt')  # Nano (fastest)
```

### Improve Accuracy
```python
# 1. Lower detection threshold
system_state.threshold = 0.3

# 2. Use larger YOLO model
models['yolo'] = YOLO('yolov8l.pt')

# 3. Higher resolution
frame = cv2.resize(frame, (1280, 960))

# 4. GPU acceleration
# Install CUDA and set torch to use GPU
```

---

## 🐛 Troubleshooting

### Issue: "Cannot open video source"
**Solution:**
- Check webcam permissions
- Try using video file instead
- Restart camera application

### Issue: "YOLO model not loaded"
**Solution:**
```bash
pip install ultralytics torch torchvision
# First run will download model automatically
```

### Issue: "face_recognition not found"
**Solution:**
```bash
pip install face-recognition dlib
# May require cmake: pip install cmake
```

### Issue: Low FPS
**Solution:**
- Reduce frame resolution in `app.py`
- Use GPU acceleration
- Lower detection threshold
- Close other applications

### Issue: High memory usage
**Solution:**
- Reduce `MAX_FRAME_BUFFER` in CONFIG
- Use smaller YOLO model (nano)
- Clear browser cache

---

## 🚀 Advanced Features

### Custom Face Database
```python
# In app.py, create a known faces database
KNOWN_FACES = {
    'person_name': face_encoding,
    'suspect_1': face_encoding,
}

# Then modify recognize_faces() function
```

### Alert Notifications
```python
# Send email/SMS on anomaly detection
import smtplib

def send_alert(message):
    # Send email notification
    pass
```

### Database Integration
```python
# Store detection results in database
from sqlalchemy import create_engine

engine = create_engine('sqlite:///detections.db')
```

### GPU Acceleration
```python
# Enable CUDA for faster processing
import torch
torch.cuda.is_available()  # Check if GPU available
```

---

## 📝 Code Examples

### Use Custom YOLO Weights
```python
# In app.py
models['yolo'] = YOLO('path/to/custom_weights.pt')
```

### Access Detection Data from Frontend
```javascript
// In console
getSystemState()  // View all system state
sendCommand({action: 'reset'})  // Reset counters
```

### Modify Alert Rules
```python
# In app.py, modify detect_anomalies()
def detect_anomalies(detections, frame_shape):
    # Add your custom logic
    if len(detections) > 150:  # If over 150 people
        return 5
    return 0
```

---

## 🎨 Customization

### Change Colors
Edit `style.css`:
```css
:root {
    --primary-color: #00d4ff;  /* Change cyan to your color */
    --secondary-color: #ff6b6b; /* Change red */
    --danger-color: #ff6b6b;
}
```

### Change Heatmap Colors
Edit `drawHeatmap()` in `script.js`:
```javascript
const hue = (1 - intensity) * 240;  // Blue to Red
// Change 240 for different color gradients
```

### Modify Dashboard Layout
Edit `style.css` grid layout:
```css
.main-grid {
    grid-template-columns: 2fr 1fr 1fr;  /* Adjust column sizes */
}
```

---

## 📦 Deployment

### Using Gunicorn (Production)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

### Cloud Deployment
- **Heroku**: Add `Procfile` with `web: gunicorn app:app`
- **AWS**: Deploy to EC2 or Lambda
- **Azure**: Use App Service

---

## 📚 Resources

- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **OpenCV**: https://opencv.org/
- **Face Recognition**: https://github.com/ageitgey/face_recognition
- **Flask Docs**: https://flask.palletsprojects.com/

---

## ⚠️ Limitations

- **CPU-based**: On CPU, may run at 15-24 FPS
- **Single camera**: Currently supports one video source
- **Face database**: Currently uses simple matching
- **No persistence**: Data cleared on restart

---

## 🔐 Security Notes

- ⚠️ This is a demonstration system
- Do not expose detection data publicly
- Implement authentication for production
- Ensure GDPR/privacy compliance for face data
- Add encryption for data at rest/in transit

---

## 📄 License

This project is provided as-is for educational and commercial use.

---

## 🤝 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the API documentation
3. Check YOLOv8/face_recognition docs

---

## 🎯 Next Steps

1. ✅ Install dependencies
2. ✅ Run `python app.py`
3. ✅ Open http://localhost:5000
4. ✅ Test with webcam or upload video
5. ✅ Customize for your needs!

---

Happy Monitoring! 🎉
