# utils/visualization.py - Visualization utilities

import matplotlib.pyplot as plt
import pandas as pd

def comparisonGraph():
    """Plot comparison graph for VGG16 and YOLOv6 performance metrics"""
    # Import metrics from model files
    from models.vgg16_model import accuracy, precision, recall, fscore
    
    df = pd.DataFrame([
        ['VGG16', 'Accuracy', accuracy[0]],
        ['VGG16', 'Precision', precision[0]],
        ['VGG16', 'Recall', recall[0]],
        ['VGG16', 'FSCORE', fscore[0]],
        ['YoloV6', 'Accuracy', accuracy[1]],
        ['YoloV6', 'Precision', precision[1]],
        ['YoloV6', 'Recall', recall[1]],
        ['YoloV6', 'FSCORE', fscore[1]],
    ], columns=['Algorithms', 'Metrics', 'Value'])
    
    df.pivot("Algorithms", "Metrics", "Value").plot(kind='bar')
    plt.rcParams["figure.figsize"] = [8, 5]
    plt.title("Algorithm Performance Comparison")
    plt.ylabel("Score (%)")
    plt.xlabel("Algorithms")
    plt.legend(title="Metrics")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_training_history(history_path, model_name):
    """
    Plot training history for a model
    
    Args:
        history_path: Path to saved training history
        model_name: Name of the model for title
    """
    import pickle
    
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title(f'{model_name} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(f'{model_name} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()

def display_detection_stats(detections):
    """
    Display detection statistics
    
    Args:
        detections: Dictionary containing detection counts by class
    """
    classes = list(detections.keys())
    counts = list(detections.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color=['red', 'blue', 'green', 'orange'])
    plt.title('Vehicle Detection Statistics')
    plt.xlabel('Vehicle Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
