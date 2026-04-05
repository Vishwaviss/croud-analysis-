"""
Configuration Module for Crowd Monitoring System
Customize these settings to match your requirements
"""

# ============================================================================
# VIDEO CAPTURE SETTINGS
# ============================================================================

VIDEO_CONFIG = {
    # Input source
    'DEFAULT_SOURCE': 0,  # 0 for webcam, or path to video file
    
    # Frame processing
    'FRAME_WIDTH': 640,
    'FRAME_HEIGHT': 480,
    'FPS': 30,
    
    # Recording (optional)
    'RECORD_VIDEO': False,
    'RECORD_PATH': 'recordings/',
    
    # Buffering
    'FRAME_BUFFER_SIZE': 30,
    'MAX_QUEUE_SIZE': 100,
}

# ============================================================================
# AI MODEL SETTINGS
# ============================================================================

AI_CONFIG = {
    # YOLO Settings
    'YOLO': {
        'MODEL': 'yolov8n.pt',  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
        'DEVICE': 'cpu',  # 'cpu' or 'cuda' (if GPU available)
        'CONFIDENCE_THRESHOLD': 0.5,  # Default threshold
        'IOU_THRESHOLD': 0.45,
        'CLASS_TO_DETECT': 0,  # 0 = person in COCO dataset
    },
    
    # Face Recognition Settings
    'FACE_RECOGNITION': {
        'ENABLED': True,
        'MODEL': 'hog',  # 'hog' (CPU) or 'cnn' (GPU, more accurate)
        'FACE_DETECTION_THRESHOLD': 0.6,
        'RECOGNITION_THRESHOLD': 0.6,  # How close faces must match
        'KNOWN_FACES_PATH': 'known_faces/',
    },
    
    # Face Recognition Model Settings
    'FACE_ENCODING': {
        'NUM_JITTERS': 1,  # More = slower but more accurate (1-10)
        'MODEL': 'small',  # 'small' or 'large'
    },
}

# ============================================================================
# DETECTION & ANALYSIS SETTINGS
# ============================================================================

DETECTION_CONFIG = {
    # People counting
    'MIN_PEOPLE_FOR_ALERT': 50,
    'HIGH_CROWD_THRESHOLD': 100,
    
    # Heatmap
    'HEATMAP_GRID_SIZE': 40,
    'HEATMAP_BLUR_SIGMA': 2.0,
    'HEATMAP_COLOR_MAP': 'hot',  # 'hot', 'cool', 'viridis', etc.
    
    # Anomaly detection
    'ANOMALY_DETECTION_ENABLED': True,
    'ANOMALY_CLUSTER_RADIUS': 50,  # Pixels
    'ANOMALY_CLUSTER_THRESHOLD': 15,  # Min people in cluster
    
    # Tracking (future feature)
    'ENABLE_TRACKING': False,
    'TRACK_HISTORY_SIZE': 50,
}

# ============================================================================
# ALERT SETTINGS
# ============================================================================

ALERT_CONFIG = {
    # Alert thresholds
    'HIGH_CROWD_THRESHOLD': 100,
    'ANOMALY_THRESHOLD': 2,
    'SUSPICIOUS_FACE_THRESHOLD': 0.95,
    
    # Alert types
    'ALERT_ON_HIGH_CROWD': True,
    'ALERT_ON_ANOMALY': True,
    'ALERT_ON_SUSPICIOUS_FACE': True,
    'ALERT_ON_SUDDEN_CHANGE': True,
    
    # Notification settings
    'MAX_ALERTS_STORED': 100,
    'ALERT_COOLDOWN': 5,  # Seconds between alerts of same type
    
    # Optional: External notifications
    'SEND_EMAIL': False,
    'EMAIL_TO': 'admin@example.com',
    'SEND_WEBHOOK': False,
    'WEBHOOK_URL': 'https://example.com/alert',
}

# ============================================================================
# DISPLAY & UI SETTINGS
# ============================================================================

DISPLAY_CONFIG = {
    # Video display
    'SHOW_BOUNDING_BOXES': True,
    'SHOW_CONFIDENCE': True,
    'SHOW_DETECTION_COUNT': True,
    'BBOX_COLOR': (0, 255, 0),  # BGR format
    'BBOX_THICKNESS': 2,
    
    # Heatmap display
    'SHOW_HEATMAP': True,
    'HEATMAP_OPACITY': 0.6,
    
    # Face display
    'SHOW_FACE_BBOX': True,
    'FACE_BBOX_COLOR': (255, 0, 0),  # Red
    
    # Performance info
    'SHOW_FPS': True,
    'SHOW_PROCESSING_TIME': True,
    
    # UI colors (RGB)
    'PRIMARY_COLOR': (0, 212, 255),  # Cyan
    'DANGER_COLOR': (255, 107, 107),  # Red
    'SUCCESS_COLOR': (81, 207, 102),  # Green
}

# ============================================================================
# DATABASE SETTINGS (Optional)
# ============================================================================

DATABASE_CONFIG = {
    'ENABLED': False,  # Set to True to enable database logging
    'DB_TYPE': 'sqlite',  # 'sqlite', 'mysql', 'postgresql'
    'DB_PATH': 'detections.db',  # For SQLite
    
    # For MySQL/PostgreSQL
    'DB_HOST': 'localhost',
    'DB_USER': 'root',
    'DB_PASSWORD': 'password',
    'DB_NAME': 'crowd_monitoring',
    
    # What to log
    'LOG_DETECTIONS': True,
    'LOG_ANOMALIES': True,
    'LOG_FACES': True,
    'LOG_PERFORMANCE': True,
    
    # Data retention (days)
    'RETENTION_DAYS': 30,
}

# ============================================================================
# SERVER SETTINGS
# ============================================================================

SERVER_CONFIG = {
    'HOST': '0.0.0.0',
    'PORT': 5000,
    'DEBUG': True,
    'THREADED': True,
    
    # CORS (Cross-Origin Resource Sharing)
    'ALLOW_CORS': True,
    'CORS_ORIGINS': ['*'],  # Restrict in production
    
    # Security
    'REQUIRE_AUTH': False,  # Set to True for authentication
    'API_KEY': 'your-secret-key-here',
    
    # Workers (for production)
    'NUM_WORKERS': 4,
    'WORKER_TIMEOUT': 120,
}

# ============================================================================
# PROCESSING PERFORMANCE SETTINGS
# ============================================================================

PERFORMANCE_CONFIG = {
    # Skip frames for faster processing
    'PROCESS_EVERY_N_FRAMES': 1,  # 1 = every frame, 2 = every 2nd frame, etc.
    
    # Multi-threading
    'NUM_THREADS': 4,
    'USE_QUEUE': True,
    
    # Memory management
    'MAX_STORED_FACES': 50,
    'MAX_STORED_DETECTIONS': 100,
    'CLEAR_CACHE_INTERVAL': 300,  # Seconds
    
    # GPU settings (if available)
    'USE_GPU': False,
    'GPU_MEMORY_FRACTION': 0.5,  # Use only 50% of GPU memory
}

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

LOGGING_CONFIG = {
    'LEVEL': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'LOG_FILE': 'crowd_monitoring.log',
    'MAX_LOG_SIZE': 10485760,  # 10MB
    'BACKUP_COUNT': 5,
    
    # What to log
    'LOG_DETECTIONS': True,
    'LOG_ERRORS': True,
    'LOG_PERFORMANCE': True,
    'LOG_API_CALLS': True,
}

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

ADVANCED_CONFIG = {
    # Face recognition features
    'BUILD_FACE_DATABASE': False,
    'KNOWN_FACES_PATH': 'known_faces/',
    'MATCH_KNOWN_FACES': False,
    
    # Advanced detection
    'USE_TRACKING': False,  # Track people across frames
    'USE_POSE_DETECTION': False,  # Detect body pose
    'USE_BEHAVIOR_ANALYSIS': False,  # Analyze crowd behavior
    
    # Custom weights
    'CUSTOM_YOLO_WEIGHTS': None,  # Path to custom YOLO weights
    
    # API rate limiting
    'RATE_LIMIT_ENABLED': True,
    'RATE_LIMIT_REQUESTS': 100,
    'RATE_LIMIT_PERIOD': 60,  # Seconds
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_config(section):
    """Get configuration section"""
    config_map = {
        'video': VIDEO_CONFIG,
        'ai': AI_CONFIG,
        'detection': DETECTION_CONFIG,
        'alert': ALERT_CONFIG,
        'display': DISPLAY_CONFIG,
        'database': DATABASE_CONFIG,
        'server': SERVER_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'logging': LOGGING_CONFIG,
        'advanced': ADVANCED_CONFIG,
    }
    return config_map.get(section, {})

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    if DETECTION_CONFIG['HEATMAP_GRID_SIZE'] < 10:
        errors.append("HEATMAP_GRID_SIZE should be at least 10")
    
    if AI_CONFIG['YOLO']['CONFIDENCE_THRESHOLD'] < 0 or \
       AI_CONFIG['YOLO']['CONFIDENCE_THRESHOLD'] > 1:
        errors.append("CONFIDENCE_THRESHOLD should be between 0 and 1")
    
    if SERVER_CONFIG['PORT'] < 1000 or SERVER_CONFIG['PORT'] > 65535:
        errors.append("PORT should be between 1000 and 65535")
    
    if errors:
        print("⚠️  Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✅ Configuration validated successfully")
    return True

def print_config(section=None):
    """Print current configuration"""
    if section:
        config = get_config(section)
        print(f"\n{section.upper()} Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        print("\nFull Configuration:")
        for section_name in ['video', 'ai', 'detection', 'alert', 'display', 
                            'database', 'server', 'performance', 'logging', 'advanced']:
            print_config(section_name)

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    # Validate and print configuration
    validate_config()
    print_config('ai')  # Print AI config
    
    # Access specific setting
    model = AI_CONFIG['YOLO']['MODEL']
    print(f"\nUsing YOLO model: {model}")
