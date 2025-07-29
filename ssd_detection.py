# detection/ssd_detection.py - SSD detection implementation

import cv2
import numpy as np
import tensorflow as tf
from config import MODEL_PATHS, CLASSES, COLORS, DETECTION_CONFIG, VIDEO_CONFIG

# Load SSD model
net = cv2.dnn.readNetFromCaffe(MODEL_PATHS['SSD_PROTOTXT'], MODEL_PATHS['SSD_MODEL'])

def ssdDetection(image_np):
    """
    Perform SSD detection on a single frame
    
    Args:
        image_np: Input image as numpy array
        
    Returns:
        Processed image with detections drawn
    """
    count = 0
    (h, w) = image_np.shape[:2]
    
    ssd = tf.Graph()
    with ssd.as_default():
        od_graphDef = tf.GraphDef()
        with tf.gfile.GFile(MODEL_PATHS['FROZEN_GRAPH'], 'rb') as file:
            serializedGraph = file.read()
            od_graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(od_graphDef, name='')
    
    with ssd.as_default():
        with tf.Session(graph=ssd) as sess:
            # Create blob from image
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image_np, DETECTION_CONFIG['input_size']),
                DETECTION_CONFIG['scale_factor'], 
                DETECTION_CONFIG['input_size'], 
                DETECTION_CONFIG['mean_subtraction']
            )
            
            # Set input to the network
            net.setInput(blob)
            detections = net.forward()
            
            # Process detections
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > DETECTION_CONFIG['confidence_threshold']:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    if (confidence * 100) > 0:
                        if CLASSES[idx] in ["bicycle", "bus", "car"]:
                            count += 1
                            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
                            
                            # Draw bounding box and label
                            cv2.rectangle(image_np, (startX, startY), (endX, endY), 
                                        COLORS[idx], 2)
                            cv2.putText(image_np, label, (startX, startY - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, 
                                      cv2.LINE_AA)
    
    # Display vehicle count
    cv2.putText(image_np, f"Detected Count : {count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
    
    return image_np

def extensionSingleShot():
    """Run SSD detection on video stream"""
    global filename
    filename = VIDEO_CONFIG['default_video']
    
    video = cv2.VideoCapture(filename)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        frame = ssdDetection(frame)
        cv2.imshow("SSD Detection", frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()
