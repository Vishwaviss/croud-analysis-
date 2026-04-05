
# 🚀 Quick Setup Guide

Choose your operating system below:

## 🪟 Windows Setup

### Step 1: Install Python
1. Download Python 3.9+ from https://www.python.org/
2. ✅ Check "Add Python to PATH" during installation
3. Open Command Prompt and verify:
   ```bash
   python --version
   pip --version
   ```

### Step 2: Create Project Folder
```bash
# Create folder
mkdir C:\crowd-monitoring
cd C:\crowd-monitoring

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# First, install cmake for dlib
pip install cmake

# Then install all requirements
pip install -r requirements.txt

# If face_recognition fails, try:
pip install face-recognition --no-deps
pip install dlib
```

### Step 4: Copy Files
Copy these files to your `crowd-monitoring` folder:
- `app.py`
- `requirements.txt`
- `README.md`
- Create folders: `templates/`, `static/`, `uploads/`
- Move `index.html` → `templates/`
- Move `style.css`, `script.js` → `static/`

### Step 5: Modify app.py
Add this to `app.py` for serving static files:

```python
from flask import send_from_directory

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)
```

And update HTML link:
```python
@app.route('/')
def index():
    return render_template('index.html')
```

Update script and CSS paths in `index.html`:
```html
<link rel="stylesheet" href="/static/style.css">
<script src="/static/script.js"></script>
```

### Step 6: Run the System
```bash
python app.py
```

Visit: **http://localhost:5000**

---

## 🍎 macOS Setup

### Step 1: Install Python
```bash
# Using Homebrew (install from https://brew.sh if needed)
brew install python3

# Or download from https://www.python.org/
```

### Step 2: Create Project
```bash
mkdir ~/crowd-monitoring
cd ~/crowd-monitoring

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install cmake first
brew install cmake

# Install requirements
pip install -r requirements.txt

# If dlib fails:
brew install dlib
pip install dlib
```

### Step 4: Copy Files
- Copy all files to `~/crowd-monitoring/`
- Create: `templates/`, `static/`, `uploads/` folders
- Move `index.html` → `templates/`
- Move `style.css`, `script.js` → `static/`

### Step 5: Modify app.py
(Same as Windows steps above)

### Step 6: Run
```bash
source venv/bin/activate
python app.py
```

Visit: **http://localhost:5000**

---

## 🐧 Linux Setup

### Step 1: Install Python
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv

# Check version
python3 --version
```

### Step 2: Create Project
```bash
mkdir ~/crowd-monitoring
cd ~/crowd-monitoring

# Virtual environment
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install System Dependencies
```bash
# Required for dlib and face_recognition
sudo apt-get install build-essential cmake git libopenblas-dev liblapack-dev libblas-dev

# For OpenCV
sudo apt-get install libsm6 libxext6 libxrender-dev
```

### Step 4: Install Python Packages
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# If dlib fails:
pip install cmake
pip install dlib
```

### Step 5: Copy Files
(Same as above)

### Step 6: Run
```bash
source venv/bin/activate
python app.py
```

---

## ✅ Verify Installation

After running `python app.py`, you should see:

```
============================================================
🎯 AI Crowd Monitoring System - Backend
============================================================
YOLO Available: True
Face Recognition Available: True

Starting server on http://localhost:5000
============================================================
```

Then open your browser and visit: **http://localhost:5000**

---

## 🐛 If You Get Errors

### Error: "ModuleNotFoundError: No module named 'flask'"
```bash
# Activate virtual environment first (if using one)
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Then install
pip install flask flask-cors
```

### Error: "dlib failed to build"
**Windows:**
```bash
# Download pre-built wheel from https://github.com/z-mahmud/dlib_prebuilt
pip install dlib-19.24-cp39-cp39-win_amd64.whl
```

**Mac/Linux:**
```bash
brew install dlib  # macOS
sudo apt-get install libdlib-dev  # Linux
pip install dlib
```

### Error: "YOLO model failed to download"
```bash
# Models download automatically first time
# Make sure you have internet connection
# Or manually download from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### Error: "Cannot open video source"
- Ensure your webcam is connected
- Check if other apps are using the camera
- Restart your computer

---

## 📱 Test the System

### 1. **Test with Webcam**
- Open dashboard
- Should see live video immediately
- Stand in front of camera
- You should see detection boxes and count

### 2. **Test with Video File**
- Prepare a video file (.mp4, .avi, etc.)
- Select "Upload Video" on dashboard
- Choose your video file
- Click "Choose Video" button
- Processing starts automatically

### 3. **Check Real-time Data**
- Open browser console (F12)
- Type: `getSystemState()`
- Should show real-time metrics

### 4. **Test Threshold**
- Move the threshold slider
- Lower = more detections (but more false positives)
- Higher = fewer detections (but more accurate)

---

## 🎯 Next: Customization

After setup works, check `README.md` for:
- Advanced configuration
- Custom AI models
- Performance optimization
- Database integration
- Cloud deployment

---

## 💡 Tips

✅ **Use virtual environment** to avoid package conflicts
✅ **Test with webcam first** before video files
✅ **Keep browser console open** to debug issues
✅ **Check port 5000 is not in use**: `python -m http.server 5000`
✅ **First YOLO run will download model** (takes 5 min)

---

## 🆘 Still Having Issues?

1. Check the full README.md
2. Check Python version (3.8+)
3. Verify all files are in correct folders
4. Try reinstalling: `pip install --upgrade -r requirements.txt`
5. Restart your computer

Good luck! 🎉
