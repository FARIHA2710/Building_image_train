import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_dir = pathlib.Path('data')
building_model = 'building_model.keras'

# Define model parameters
epochs = 100
batch_size = 10
img_height = 1000
img_width = 1200

# Load data using a Keras utility
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
logging.info(f"Classes found: {class_names}")

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Define the model with data augmentation and dropout
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),  # Dropout to prevent overfitting
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

if os.path.exists(building_model):
    logging.info(f"Initiating model retrieval from storage: '{building_model}'")
    model = tf.keras.models.load_model(building_model)
else:
    model.summary()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save(building_model)

    # Visualize training results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def classify_image(image_path, expected_label):
    img = tf.keras.utils.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_names[predicted_index]
    confidence = np.max(tf.nn.softmax(predictions[0]))

    logging.info(
        f"'{image_path}' is classified as {predicted_label} (confidence: {confidence * 100:.2f}%)"
    )
    return predicted_label == expected_label

def evaluate_directory(directory, file_extensions, expected_label):
    correct_count = 0
    total_count = 0
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(tuple(file_extensions)):
                file_path = os.path.join(subdir, file)
                if classify_image(file_path, expected_label):
                    correct_count += 1
                total_count += 1
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    logging.info(
        f"Accuracy for '{directory}': {accuracy:.2f}% ({correct_count}/{total_count})"
    )

evaluate_directory('data/TYPE_1464', ['.jpg'], 'TYPE_1464')
evaluate_directory('data/OTHER', ['.jpg'], 'OTHER')
