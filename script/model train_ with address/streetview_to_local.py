import os
import requests
import logging
from PIL import Image
from io import BytesIO
from typing import Dict, Union, Optional, List
from addresses import type_1464_addresses, other_type_addresses

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_KEY = os.getenv("API_KEY")
                    
def get_streetview(
    address: str,
    width: int = 1024,
    height: int = 1024,
    fov: int = 120,
    pitch: int = 0,
) -> Optional[Image.Image]:
    """
    Fetches a street view image for the given address using the Google Street View API.

    Args:
        address (str): The address of the location.
        api_key (str): Your Google Street View API key.
        width (int): Image width (default 640, can be set up to 1024 for high-resolution images).
        height (int): Image height (default 640, can be set up to 1024 for high-resolution images).
        fov (int): Field of view of the camera (default 120, range 0 to 120).
        pitch (int): The up or down angle of the camera relative to the street level (default 0).

    Returns:
        PIL.Image.Image or None: Returns the image if successful, None otherwise.
    """

    url = "https://maps.googleapis.com/maps/api/streetview"
    params: Dict[str, Union[str, int]] = {
        "size": f"{width}x{height}",
        "location": address,
        "fov": fov,
        "pitch": pitch,
        "key": API_KEY,
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        return img
    else:
        logging.error(
            f"Failed to fetch image for {address}. Error code: {response.status_code}"
        )
        return None

def store_streetview_images(addresses: List[str], folder_path: str):
    for address in addresses:
        image = get_streetview(address)
        if image:
            filename = f"{sanitize_filename(address)}.jpg"
            full_path = os.path.join(folder_path, filename)
            image.save(full_path)
            logging.info(f"Image saved: {full_path}")

def sanitize_filename(address: str) -> str:
    return "".join(char for char in address if char.isalnum() or char in " -_").rstrip()


# Create subdirectories if they do not exist
os.makedirs(os.path.join("data", "1-464"), exist_ok=True)
os.makedirs(os.path.join("data", "121-E"), exist_ok=True)
#print(os.path.join("data", "1-464")
# Save images to the appropriate subdirectories
store_streetview_images(type_1464_addresses, os.path.join("data", "1-464"))
store_streetview_images(other_type_addresses, os.path.join("data", "121-E"))
