# -*- coding: utf-8 -*-
"""
Created on Sat May 24 21:45:44 2025

@author: andre
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 24 20:38:45 2025

@author: andre
"""

from keras.layers import PReLU
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.applications import InceptionV3
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import tensorflow as tf
import pathlib

K.set_image_data_format('channels_last')

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')

seed = 42
np.random.seed(1)
random.seed(2)
tf.random.set_seed(3)

height, width, channels = 224, 224, 3
path = 'C:\\Anul 4\\Sem I\\ML\\'
image_path = 'MyDataset'
epochs = 30
BS = 32

def opencv_augment(image):
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
    if random.random() < 0.5:
        angle = random.randint(-20, 20)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))
    if random.random() < 0.5:
        brightness = random.uniform(0.7, 1.3)
        image = np.clip(image * brightness, 0, 255).astype(np.uint8)
    return image

def create_dataset(image_path, augment=False):
    images, labels = [], []
    for subdir, dirs, files in os.walk(image_path):
        for file in files:
            if subdir != image_path and pathlib.Path(file).suffix == '.jpg':
                image = cv2.imread(os.path.join(subdir, file))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (height, width))
                if augment:
                    image = opencv_augment(image)
                images.append(image)
                labels.append(os.path.basename(subdir))
    return np.array(images), np.array(labels)

X, Y = create_dataset(path + image_path, augment=True)

plt.figure(figsize=(10, 3))
for i in range(0, 3):
    plt.subplot(1, 3, i+1)
    plt.imshow(X[i])
    plt.title(Y[i])
    plt.axis("off")
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, stratify=Y, random_state=seed)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

encoder = LabelEncoder()
encoder.fit(Y_train)
Y_OHE_train = to_categorical(encoder.transform(Y_train))
Y_OHE_test = to_categorical(encoder.transform(Y_test))
num_classes = Y_OHE_test.shape[1]

def Inception_model():
    base = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base.layers:
        layer.trainable = False

    x = Flatten()(base.output)
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

    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.inputs, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00001), metrics=['accuracy'])

    return model

Inception_model = Inception_model()

history_inception = Inception_model.fit(
    X_train, Y_OHE_train,
    validation_data=(X_test, Y_OHE_test),
    epochs=epochs,
    batch_size=BS
)

def evaluate_model(model, X_test, Y_OHE_test, encoder, model_name="Model"):
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(Y_OHE_test, axis=1)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    class_names = encoder.classes_
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

evaluate_model(Inception_model, X_test, Y_OHE_test, encoder, model_name="InceptionV3")

def plot_training_history(history, model_name="Model"):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linestyle='--')
    plt.title(f'{model_name} - Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='green')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red', linestyle='--')
    plt.title(f'{model_name} - Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_training_history(history_inception, model_name="InceptionV3")

def predict_folder(folder_path, model, encoder, image_size=(224, 224)):
    predictions = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if pathlib.Path(file).suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img_path = os.path.join(root, file)

                image = cv2.imread(img_path)
                if image is None:
                    print(f"[ERROR] Could not read image: {img_path}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, image_size)
                image = image.astype('float32') / 255.0
                image = np.expand_dims(image, axis=0)

                pred = model.predict(image)
                predicted_class = np.argmax(pred, axis=1)[0]
                class_label = encoder.inverse_transform([predicted_class])[0]

                print(f"Image: {img_path}")
                print(f"Predicted Class: {class_label}")
                print("-" * 40)

                predictions.append((img_path, class_label))

    print(f"\n✅ Done! {len(predictions)} images predicted.")
    return predictions


# Example usage
test_folder_path = 'C:\\Anul 4\\Sem I\\ML\\MyDataset\\test\\normal'
all_predictions = predict_folder(test_folder_path, Inception_model, encoder)

save_path = os.path.join(path, 'UI-playground', 'inception_fishingnet_model.h5')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
Inception_model.save(save_path)
print(f"✅ Model saved to: {save_path}")
