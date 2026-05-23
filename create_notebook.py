import json

def create_code_cell(source_code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_code.splitlines()]
    }

cells = []

# Cell 0: Imports
cells.append(create_code_cell("""import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50, InceptionV3
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import img_to_array, load_img"""))

# Cell 1: Data Generators (with Augmentation)
cells.append(create_code_cell("""# Using Kaggle Path
data_dir = '/kaggle/input/banana-disease-recognition-dataset/Banana Disease Recognition Dataset/Augmented images/Augmented images'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Advanced Data Augmentation for Training
datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Only rescaling for validation
datagen_val = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen_train.flow_from_directory(
    data_dir, 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='categorical', 
    subset='training'
)

val_generator = datagen_val.flow_from_directory(
    data_dir, 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='categorical', 
    subset='validation',
    shuffle=False
)"""))

# Cell 2: Class Weights
cells.append(create_code_cell("""# Compute class weights to handle any class imbalance
class_weights = class_weight.compute_class_weight(
    'balanced', 
    classes=np.unique(train_generator.classes), 
    y=train_generator.classes
)
class_weights = {i : class_weights[i] for i in range(len(class_weights))}
print("Class weights:", class_weights)"""))

# Cell 3: Callbacks Definition
cells.append(create_code_cell("""# Define callbacks for robust training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]"""))

# Cell 4: Custom CNN (LeNet)
cells.append(create_code_cell("""# 1. Custom CNN Model
model_custom = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    Conv2D(256, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(256, kernel_size=(3, 3), activation='relu'),
    GlobalAveragePooling2D(),
    Dropout(0.2),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(7, activation='softmax')
])

model_custom.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model_custom.summary()"""))

# Cell 5: Train Custom CNN
cells.append(create_code_cell("""print("Training Custom CNN...")
history_custom = model_custom.fit(
    train_generator, 
    epochs=50, 
    validation_data=val_generator, 
    class_weight=class_weights,
    callbacks=callbacks
)"""))

# Cell 6: ResNet50 Definition
cells.append(create_code_cell("""# 2. ResNet50 Model (Deep, Powerful)
base_model_res = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Fine-tuning: Unfreeze the top layers
base_model_res.trainable = True
for layer in base_model_res.layers[:-20]: # Freeze all but last 20 layers
    layer.trainable = False

model_res = Sequential([
    base_model_res,
    GlobalAveragePooling2D(),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(7, activation='softmax')
])

# Use lower learning rate for fine-tuning
model_res.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model_res.summary()"""))

# Cell 7: Train ResNet50
cells.append(create_code_cell("""print("Training ResNet50...")
history_res = model_res.fit(
    train_generator, 
    epochs=50, 
    validation_data=val_generator, 
    class_weight=class_weights,
    callbacks=callbacks
)"""))

# Cell 8: InceptionV3 Definition
cells.append(create_code_cell("""# 3. InceptionV3 Model
base_model_inc = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Fine-tuning: Unfreeze the top layers
base_model_inc.trainable = True
for layer in base_model_inc.layers[:-30]: # Freeze all but last 30 layers
    layer.trainable = False

model_inc = Sequential([
    base_model_inc,
    GlobalAveragePooling2D(),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(7, activation='softmax')
])

# Use lower learning rate for fine-tuning
model_inc.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model_inc.summary()"""))

# Cell 9: Train InceptionV3
cells.append(create_code_cell("""print("Training InceptionV3...")
history_inc = model_inc.fit(
    train_generator, 
    epochs=50, 
    validation_data=val_generator, 
    class_weight=class_weights,
    callbacks=callbacks
)"""))

# Cell 10: Performance Plotting Function
cells.append(create_code_cell("""def plot_performance(history, title):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title(f'{title} Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(f'{title} Loss')
    plt.legend()

    plt.show()

def evaluate_model(model, generator):
    y_true = generator.classes
    y_pred = model.predict(generator)
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    cr = classification_report(y_true, y_pred_classes, target_names=generator.class_indices.keys())

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=generator.class_indices.keys(), yticklabels=generator.class_indices.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    print('Classification Report:\\n', cr)"""))

# Cell 11: Plot and Evaluate Models
cells.append(create_code_cell("""# Custom CNN Eval
plot_performance(history_custom, 'Custom CNN')
evaluate_model(model_custom, val_generator)

# ResNet50 Eval
plot_performance(history_res, 'ResNet50')
evaluate_model(model_res, val_generator)

# InceptionV3 Eval
plot_performance(history_inc, 'InceptionV3')
evaluate_model(model_inc, val_generator)"""))

# Cell 12: Save Models
cells.append(create_code_cell("""model_custom.save('model_custom.h5')
model_res.save('model_resnet.h5')
model_inc.save('model_inception.h5')"""))

# Cell 13: Voting Classifier
cells.append(create_code_cell("""def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size) 
    img_array = img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0 
    return img_array

class_mapping = {v: k for k, v in train_generator.class_indices.items()}

# Function to predict using each model and aggregate predictions
def predict_with_voting(image_path):
    img = preprocess_image(image_path)
    
    # Step 1: Make predictions using each model
    pred_custom = model_custom.predict(img)
    pred_res = model_res.predict(img)
    pred_inc = model_inc.predict(img)
    
    # Step 2: Aggregate predicted probabilities (soft voting)
    # Give slightly more weight to the pre-trained models as they usually perform better
    final_pred_prob = (pred_custom * 0.2 + pred_res * 0.4 + pred_inc * 0.4) 

    # Step 3: Predict class
    final_pred_class = np.argmax(final_pred_prob, axis=1)
    
    # Map index to disease
    final_pred_disease = [class_mapping[class_index] for class_index in final_pred_class]

    return final_pred_disease[0]

# Example Usage (Uncomment and edit path to test):
# image_path = '/kaggle/input/banana-disease-recognition-dataset/.../image.jpg'
# predicted_class = predict_with_voting(image_path)
# print("Predicted Class:", predicted_class)"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('improved_banana_disease_model.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)
print("Improved notebook created successfully!")
