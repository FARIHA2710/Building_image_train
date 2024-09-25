import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model_path = r'C:\Users\farih\TTU\Github\Building_image_train\Data\model\building_model.keras'
model = load_model(model_path)

# Define image properties
img_height, img_width = 1000, 1200  # these should match the input shape your model expects

# Define directories
test_dir = r'C:\Users\farih\TTU\Github\Building_image_train\Data\images\test'  # Your test images directory
sort_dir = r'C:\Users\farih\TTU\Github\Building_image_train\Data\images\sort'  # Directory to store sorted images

# Create subdirectories for each class
class_names = ['1-464', '121-E']
for class_name in class_names:
    os.makedirs(os.path.join(sort_dir, class_name), exist_ok=True)

def classify_and_move(image_path):
    """Classify the image and move it to the corresponding folder."""
    # Load and prepare the image
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

    # Make predictions
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]

    # Move the image to the corresponding directory
    destination = os.path.join(sort_dir, predicted_class, os.path.basename(image_path))
    os.rename(image_path, destination)
    print(f'Moved {image_path} -> {destination}')

# Process all images in the test directory
for filename in os.listdir(test_dir):
    file_path = os.path.join(test_dir, filename)
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
        classify_and_move(file_path)

print("Image sorting complete.")
