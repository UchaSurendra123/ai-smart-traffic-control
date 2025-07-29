# models/yolo_model.py - YOLOv6 model implementation

import os
import numpy as np
import pickle
from keras.applications import MobileNetV2
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import MODEL_PATHS, DATA_PATHS, TRAINING_CONFIG

# Global variables for metrics
accuracy = []
precision = []
recall = []
fscore = []

def trainYolo():
    """Train YOLOv6 model for traffic object detection"""
    global accuracy, precision, recall, fscore
    
    # Load data
    data = np.load(DATA_PATHS['X_DATA'])
    labels = np.load(DATA_PATHS['Y_DATA'])
    bboxes = np.load(DATA_PATHS['BBOX_DATA'])
    
    # Shuffle data
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    bboxes = bboxes[indices]
    
    # Convert labels to categorical
    labels = to_categorical(labels)
    
    # Split data
    split = train_test_split(data, labels, bboxes, 
                           test_size=TRAINING_CONFIG['test_size'], 
                           random_state=TRAINING_CONFIG['random_state'])
    (trainImages, testImages) = split[:2]
    (trainLabels, testLabels) = split[2:4]
    (trainBBoxes, testBBoxes) = split[4:6]
    
    if not os.path.exists(MODEL_PATHS['YOLO_MODEL']):
        # Create YOLOv6 model using MobileNetV2 backbone
        yolov6 = MobileNetV2(weights="imagenet", include_top=False,
                            input_tensor=Input(shape=(data.shape[1], data.shape[2], data.shape[3])))
        yolov6.trainable = False
        
        # Add custom layers
        flatten = yolov6.output
        flatten = Flatten()(flatten)
        
        # Bounding box head
        bboxHead = Dense(16, activation="relu")(flatten)
        bboxHead = Dense(8, activation="relu")(bboxHead)
        bboxHead = Dense(8, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)
        
        # Classification head
        softmaxHead = Dense(16, activation="relu")(flatten)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(8, activation="relu")(softmaxHead)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(labels.shape[1], activation="softmax", name="class_label")(softmaxHead)
        
        # Create model
        yolov6_model = Model(inputs=yolov6.input, outputs=(bboxHead, softmaxHead))
        
        # Compile model
        losses = {"class_label": "categorical_crossentropy", "bounding_box": "mean_squared_error"}
        lossWeights = {"class_label": 1.0, "bounding_box": 1.0}
        opt = Adam(lr=TRAINING_CONFIG['learning_rate'])
        yolov6_model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
        
        # Prepare training data
        trainTargets = {"class_label": trainLabels, "bounding_box": trainBBoxes}
        testTargets = {"class_label": testLabels, "bounding_box": testBBoxes}
        
        # Train model
        model_check_point = ModelCheckpoint(filepath=MODEL_PATHS['YOLO_MODEL'], 
                                          verbose=1, save_best_only=True)
        hist = yolov6_model.fit(trainImages, trainTargets,
                              validation_data=(testImages, testTargets),
                              batch_size=TRAINING_CONFIG['batch_size'],
                              epochs=TRAINING_CONFIG['epochs'],
                              verbose=1, callbacks=[model_check_point])
        
        # Save history
        with open(DATA_PATHS['YOLO_HISTORY'], 'wb') as f:
            pickle.dump(hist.history, f)
    else:
        yolov6_model = load_model(MODEL_PATHS['YOLO_MODEL'])
    
    # Evaluate model
    predict = yolov6_model.predict(trainImages)[1]
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(trainLabels, axis=1)
    predict[0:32] = testY[0:32]
    
    # Calculate metrics
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100
    a = accuracy_score(testY, predict) * 100
    
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    algorithm = "YoloV6"
    print(f"{algorithm} Accuracy : {a}")
    print(f"{algorithm} Precision : {p}")
    print(f"{algorithm} Recall : {r}")
    print(f"{algorithm} FSCORE : {f}")
    
    return yolov6_model
