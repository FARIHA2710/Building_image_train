import pathlib
import subprocess
import sys
import re

def get_system_path(*args):
    """Utility function to create system-dependent paths."""
    return pathlib.Path(*args)

def run_command(command):
    """Utility function to run a command as a subprocess and capture its output."""
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode == 0:
        print("Command executed successfully!")
        print("Output:", process.stdout)
        return process.stdout
    else:
        print("Error in command execution!")
        print("Error:", process.stderr)
        return None

def main():
    print("Working")
    weights_path = get_system_path(r'C:\Users\farih\TTU\Github\Building_image_train\script\facade_test_yolov5\yolov5\best3class1050.pt')
    source_image_path = get_system_path(r'C:\Users\farih\TTU\Github\Building_image_train\script\facade_test_yolov5\yolov5\test\1-317 _101020735.jpg')
    model1 = r'C:\Users\farih\TTU\Github\Building_image_train\script\facade_test_yolov5\yolov5\detect_building_type.py'
    model2 = r'C:\Users\farih\TTU\Github\Building_image_train\script\facade_test_yolov5\yolov5\detect3.py'

    # Run the first model
    command1 = [
        sys.executable,
        model1,
        '--weights', str(weights_path),
        '--line-thickness', '1',
        '--conf-thres', '0.15',
        '--source', str(source_image_path)
    ]
    output1 = run_command(command1)

    if output1:
        if "317" in output1:
            print("Detected building type 317, No subtype assinged so, stopping the process.")
            return
        elif "464" in output1 or "121" in output1:
            # Extract the directory path from the output using regular expressions
            path_match = re.search(r"Image saved to (.+\.png)", output1)
            if path_match:
                output_directory = str(pathlib.Path(path_match.group(1)).parent)

                # Prepare and run the second model
                second_model_weights = get_system_path(r'C:\Users\farih\TTU\Github\Building_image_train\script\facade_test_yolov5\yolov5\door_window_detector.pt')
                second_model_command = [
                    sys.executable,
                    model2,
                    '--weights', str(second_model_weights),
                    '--conf-thres', '0.15',  # Adjust thresholds as necessary
                    '--source', str(source_image_path),
                    '--output', output_directory  # Use the extracted directory
                ]
                run_command(second_model_command)
            else:
                print("Could not find the path in the output.")
        else:
            print("No relevant building type detected.")

if __name__ == "__main__":
    main()



# import pathlib
# import subprocess
# import sys
# import re
#
# def get_system_path(*args):
#     """Utility function to create system-dependent paths."""
#     return pathlib.Path(*args)
#
# def run_command(command):
#     """Utility function to run a command as a subprocess and capture its output."""
#     process = subprocess.run(command, capture_output=True, text=True)
#     if process.returncode == 0:
#         print("Command executed successfully!")
#         print("Output:", process.stdout)
#         return process.stdout
#     else:
#         print("Error in command execution!")
#         print("Error:", process.stderr)
#         return None
#
# def main():
#     print("Working")
#     weights_path = get_system_path(r'C:\Users\farih\TTU\Github\Building_image_train\script\facade_test_yolov5\yolov5\best3class1050.pt')
#     source_image_path = get_system_path(r'C:\Users\farih\TTU\Github\Building_image_train\script\facade_test_yolov5\yolov5\test\464 evilde 121.png')
#     model = r'C:\Users\farih\TTU\Github\Building_image_train\script\facade_test_yolov5\yolov5\detect_building_type.py'
#
#     # Run the first model
#     command1 = [
#         sys.executable,
#         model,
#         '--weights', str(weights_path),
#         '--line-thickness', '1',
#         '--conf-thres', '0.15',
#         '--source', str(source_image_path)
#     ]
#     output1 = run_command(command1)
#
#     # Extract the directory path from the output using regular expressions
#     if output1:
#         path_match = re.search(r"Image saved to (.+\.png)", output1)
#         if path_match:
#             output_directory = str(pathlib.Path(path_match.group(1)).parent)
#
#             # Path for the second model
#             model2 = r'C:\Users\farih\TTU\Github\Building_image_train\script\facade_test_yolov5\yolov5\detect3.py'
#             second_model_weights = get_system_path(r'C:\Users\farih\TTU\Github\Building_image_train\script\facade_test_yolov5\yolov5\door_window_detector.pt')
#             second_model_command = [
#                 sys.executable,
#                 model2,
#                 '--weights', str(second_model_weights),
#                 '--conf-thres', '0.15',  # Adjust thresholds as necessary
#                 '--source', str(source_image_path),
#                 '--output', output_directory  # Use the extracted directory
#             ]
#             run_command(second_model_command)
#         else:
#             print("Could not find the path in the output.")
#
# if __name__ == "__main__":
#     main()

#
#
# import pathlib
# import subprocess
# import sys
#
# def get_system_path(*args):
#     """Utility function to create system-dependent paths."""
#     return pathlib.Path(*args)
#
# def run_command(command):
#     """Utility function to run a command as a subprocess."""
#     process = subprocess.run(command, capture_output=True, text=True)
#     if process.returncode == 0:
#         print("Command executed successfully!")
#         print("Output:", process.stdout)
#         return process.stdout
#     else:
#         print("Error in command execution!")
#         print("Error:", process.stderr)
#         return None
#
# def main():
#     print("Working")
#     weights_path = get_system_path(r'C:\Users\farih\TTU\Github\Building_image_train\script\facade_test_yolov5\yolov5\best3class1050.pt')
#     source_image_path = get_system_path(r'C:\Users\farih\TTU\Github\Building_image_train\script\facade_test_yolov5\yolov5\test\464 evilde 121.png')
#     model = r'C:\Users\farih\TTU\Github\Building_image_train\script\facade_test_yolov5\yolov5\detect_building_type.py'
#     model2 = r'C:\Users\farih\TTU\Github\Building_image_train\script\facade_test_yolov5\yolov5\detect3.py'
#
#     # Run the first model
#     command = [
#         sys.executable,
#         model,
#         '--weights', str(weights_path),
#         '--line-thickness', '1',
#         '--conf-thres', '0.15',
#         '--source', str(source_image_path)
#     ]
#     output = run_command(command)
#
#     # Assuming the output contains some recognizable pattern or is JSON-formatted
#     if output and ("464" in output or "121" in output):
#         # Conditions to run the second model
#         print("Running second model for detected building type.")
#         second_model_weights = get_system_path(r'C:\Users\farih\TTU\Github\Building_image_train\script\facade_test_yolov5\yolov5\door_window_detector.pt')
#         second_model_command = [
#             sys.executable,
#             model2,  # Assuming the same script can be used or point to a different script
#             '--weights', str(second_model_weights),
#             '--conf-thres', '0.15',  # Adjust thresholds as necessary
#             '--source', str(source_image_path)
#         ]
#         run_command(second_model_command)
#     elif output and "317" in output:
#         print("Detected building type 317, stopping the process.")
#
# if __name__ == "__main__":
#     main()
