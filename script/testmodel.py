import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Load the trained model
model_path = r"C:\Users\farih\TTU\Github\Building_image_train\model3"  # Update this to the path where your model is saved
model = load_model(model_path)

# Define the directory containing the test images
test_images_directory = r"C:\Users\farih\TTU\Github\Building_image_train\Data\images\test" # Update this to the path of your test images
sorted_images_directory = r"C:\Users\farih\TTU\Github\Building_image_train\Data\images\sorted"  # Update this to where you want to save sorted images

# Define your class names based on the folder names from the training set
class_names = [
    '1-464',
    '1-464D',
    '1-464A',
    '111-133 (Tartu type)',
   '121-E',
   '121 (5-',
   '121 (9-'
]  # Update these class names to match the folders in your training set

# Create target directories for each class
for class_name in class_names:
    os.makedirs(os.path.join(sorted_images_directory, class_name), exist_ok=True)

# Function to preprocess images
def preprocess_image(image_path, target_size=(256, 256)):
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array


# Loop over each image in the test images directory
for filename in os.listdir(test_images_directory):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):  # Check for image files
        # Full path to the current image file
        image_path = os.path.join(test_images_directory, filename)

        # Preprocess the image
        image_array = preprocess_image(image_path)

        # Make a prediction
        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class_index]

        # Path to the target directory for the predicted class
        target_directory = os.path.join(sorted_images_directory, predicted_class_name)

        # Move the image file to the target directory
        os.rename(image_path, os.path.join(target_directory, filename))

print("Images have been sorted into respective folders based on predictions.")

