

import pathlib
pathlib.PosixPath = pathlib.WindowsPath


import sys
import pathlib
import subprocess
print("Working")
# Use pathlib.Path which automatically handles path instantiation based on the OS
def get_system_path(*args):
    """Utility function to create system-dependent paths"""
    return pathlib.Path(*args)
#
# # Example usage of the get_system_path function
weights_path = get_system_path('yolov5/door_window_detector.pt')
source_image_path = get_system_path('yolov5/test', '464 tammsaare_71.jpg')
#Define the command as a list of elements
command = [
    sys.executable,  # Use the Python interpreter from the current environment
    'yolov5/detect3.py',
    '--weights', str(weights_path),  # Convert Path object to string
    '--line-thickness', '1',
    '--conf-thres', '0.15',
    '--source', str(source_image_path)  # Convert Path object to string
]
#Define the command with additional flags for saving results



# Run the command
process = subprocess.run(command, capture_output=True, text=True)

# Check if the process was successful
if process.returncode == 0:
    print("Command executed successfully!")
    print("So Output:", process.stdout)
else:
    print("Error in command execution!")
    print("Error:", process.stderr)
print("done")