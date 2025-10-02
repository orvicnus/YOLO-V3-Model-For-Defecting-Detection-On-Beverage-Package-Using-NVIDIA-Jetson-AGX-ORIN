# check the number of images and labels
print(len(os.listdir(image_train_dir)))
print(len(os.listdir(label_train_dir)))
print(len(os.listdir(image_test_dir)))
print(len(os.listdir(label_test_dir)))

import os
import numpy as np

def check_yolo_format(file_path):
    """
    Check if a YOLO format label file is correct.

    Args:
    - file_path (str): Path to the label file.

    Returns:
    - bool: True if all labels in the file are in YOLO format, False otherwise.
    """
    try:
        # Load the data from the label file
        data = np.loadtxt(file_path, delimiter=" ", ndmin=2)

        # Check if each label in the file follows the YOLO format
        for line in data:
            if len(line) != 5:
                print(f"Error: Line {line} in {file_path} does not have 5 elements.")
                return False

            class_id, x_center, y_center, width, height = line

            if not (0 < x_center <= 1 and 0 < y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
                print(f"Error: Line {line} in {file_path} has values out of range [0, 1].")
                return False

            if not isinstance(x_center, float) or not isinstance(y_center, float) or \
              not isinstance(width, float) or not isinstance(height, float):
                print(f"Error: Line {line} in {file_path} contains non-float values.")
                return False

        return True

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return False

def check_all_yolo_labels(folder_path):
    """
    Check all YOLO format label files in a specified folder.

    Args:
    - folder_path (str): Path to the folder containing label files.
    """
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            print(f"Checking file: {file_path}")
            if check_yolo_format(file_path):
                print(f"The file {file_path} is in YOLO format.")
            else:
                print(f"The file {file_path} is not in YOLO format.")

# Check YOLO labels in training and testing directories
print("Checking training labels...")
check_all_yolo_labels(label_train_dir)

print("Checking testing labels...")
check_all_yolo_labels(label_test_dir)

#Declare The Label
class_names = ['UltraMilk_Layak', 'UltraMilk_Rusak', 'FrisianFlag_Layak', 'FrisianFlag_Rusak', 'TehKotak_Layak', 'TehKotak_Rusak']