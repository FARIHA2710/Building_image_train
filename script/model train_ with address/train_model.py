import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the environment variable to suppress oneDNN optimization messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define the paths to your training data
train_data_path = r"C:\Users\farih\TTU\Github\Building_image_train\Data\images\train"

# Set some parameters
batch_size = 32
image_height = 256
image_width = 256
num_classes = 3  # Update this to the number of building types you have

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Load and preprocess images from the training directory
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='sparse')  # 'sparse' if you have integer labels

# Define the CNN model architecture
model = Sequential([
    Rescaling(1./255, input_shape=(image_height, image_width, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=20)  # The number of times to iterate over the entire dataset

# Save the model
model_save_path = r"C:\Users\farih\TTU\Github\Building_image_train\model5"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
