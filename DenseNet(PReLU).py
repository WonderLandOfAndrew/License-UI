# -*- coding: utf-8 -*-
"""
Created on Thu May 29 08:40:26 2025

@author: andre
"""

import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pathlib

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, PReLU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

K.set_image_data_format('channels_last')

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# Parameters
height, width, channels = 224, 224, 3
path = 'C:\\Anul 4\\Sem I\\ML\\'
image_path = 'MyDataset'

def augment_image(image):
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    if random.random() > 0.5:
        angle = random.randint(-15, 15)
        M = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
        image = cv2.warpAffine(image, M, (width, height))
    if random.random() > 0.5:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    return image

def create_dataset(image_path):
    images = []
    labels = []
    for subdir, dirs, files in os.walk(image_path):
        for file in files:
            if subdir != image_path and pathlib.Path(file).suffix == '.jpg':
                image = cv2.imread(os.path.join(subdir, file))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (height, width))
                images.append(image)
                labels.append(os.path.basename(subdir))
    return images, labels

X, Y = create_dataset(path + image_path)
X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, stratify=Y, random_state=seed)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

encoder = LabelEncoder()
Y_train_encoded = encoder.fit_transform(Y_train)
Y_test_encoded = encoder.transform(Y_test)
Y_OHE_train = to_categorical(Y_train_encoded)
Y_OHE_test = to_categorical(Y_test_encoded)
num_classes = Y_OHE_test.shape[1]

def custom_generator(X, Y, batch_size):
    while True:
        idx = np.random.choice(len(X), batch_size)
        batch_X = X[idx]
        batch_Y = Y[idx]
        augmented_X = np.array([augment_image(img) for img in batch_X])
        yield augmented_X, batch_Y

def DenseNet121_model():
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(512)(x)
    x = PReLU()(x)
    x = Dropout(0.3)(x)
    x = Dense(256)(x)
    x = PReLU()(x)
    x = Dropout(0.4)(x)
    x = Dense(128)(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.inputs, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model

model = DenseNet121_model()
epochs = 30
BS = 16

history = model.fit(
    custom_generator(X_train, Y_OHE_train, BS),
    steps_per_epoch=len(X_train) // BS,
    validation_data=(X_test, Y_OHE_test),
    epochs=epochs
)

plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(Y_OHE_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - DenseNet121 with PReLU")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=encoder.classes_))

predictions = []
for root, _, files in os.walk('C:\\Anul 4\\Sem I\\ML\\MyDataset\\test\\normal'):
    for file in files:
        if pathlib.Path(file).suffix.lower() in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(root, file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"[ERROR] Could not read image: {img_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            image = image.astype('float32') / 255.0
            image = np.expand_dims(image, axis=0)

            pred = model.predict(image)
            predicted_class = np.argmax(pred, axis=1)[0]
            class_label = encoder.inverse_transform([predicted_class])[0]

            print(f"Image: {img_path}")
            print(f"Predicted Class: {class_label}")
            print("-" * 40)

            predictions.append((img_path, class_label))

# Previous code
print(f"\n✅ Done! {len(predictions)} images predicted.")

save_path = os.path.join(path, 'UI-playground', 'densenet_fishingnet_model.h5')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save(save_path)
print(f"✅ Model saved to: {save_path}")


