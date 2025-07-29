# config.py - Configuration settings for the traffic control system

import numpy as np

# Model paths
MODEL_PATHS = {
    'SSD_PROTOTXT': 'models/MobileNetSSD_deploy.prototxt.txt',
    'SSD_MODEL': 'models/MobileNetSSD_deploy.caffemodel',
    'VGG16_MODEL': 'models/vgg_model.hdf5',
    'YOLO_MODEL': 'models/yolov6.hdf5',
    'FROZEN_GRAPH': 'models/frozen_inference_graph.pb'
}

# Data paths
DATA_PATHS = {
    'X_DATA': 'models/X.txt.npy',
    'Y_DATA': 'models/Y.txt.npy',
    'BBOX_DATA': 'models/bb.txt.npy',
    'VGG_HISTORY': 'models/vgg_history.pckl',
    'YOLO_HISTORY': 'models/yolov6.pckl'
}

# Detection classes for SSD
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

# Random colors for visualization
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Detection parameters
DETECTION_CONFIG = {
    'confidence_threshold': 0.2,
    'nms_threshold': 0.4,
    'input_size': (300, 300),
    'scale_factor': 0.007843,
    'mean_subtraction': 127.5
}

# Training parameters
TRAINING_CONFIG = {
    'test_size': 0.20,
    'random_state': 42,
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 1e-4
}

# Video processing
VIDEO_CONFIG = {
    'default_video': 'Videos/traffic2.mp4',
    'output_fps': 30,
    'display_window': 'Traffic Detection'
}
