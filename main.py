# main.py - Entry point for Smart Traffic Light Control System

from traffic_simulation import *
from yolo_traffic import *
from models.vgg16_model import trainVGG16
from models.yolo_model import trainYolo
from detection.ssd_detection import ssdDetection, extensionSingleShot
from detection.yolo_detection import yoloTrafficDetection
from utils.visualization import comparisonGraph
import cv2
import numpy as np

# Global variables
global filename, accuracy, precision, recall, fscore

def main():
    """Main function to run the smart traffic control system"""
    print("Smart Traffic Light Control System - Starting...")
    
    # Train models
    print("Training VGG16 model...")
    trainVGG16()
    
    print("Training YOLOv6 model...")
    trainYolo()
    
    # Run traffic detection
    print("Running SSD detection...")
    extensionSingleShot()
    
    print("Running YOLO detection...")
    yoloTrafficDetection()
    
    # Show comparison results
    print("Generating comparison graph...")
    comparisonGraph()

if __name__ == "__main__":
    main()
