import shutil
from ultralytics import YOLO
import os

model = YOLO(r'D:\Algo Orange\yolo_training_model\runs\detect\train4\weights\best.pt')

# Export - files go to default folder
model.export(format='tflite')

# Define paths
default_export_folder = r'D:\Algo Orange\yolo_training_model\runs\detect\train4\weights\best_saved_model'
new_export_folder = r'D:\Algo Orange\yolo_training_model\runs\detect\train4\weights\tflite_export'

# Create new folder if it doesn't exist
os.makedirs(new_export_folder, exist_ok=True)

# Move or copy all generated files to the new folder
for file_name in os.listdir(default_export_folder):
    source = os.path.join(default_export_folder, file_name)
    destination = os.path.join(new_export_folder, file_name)
    shutil.move(source, destination)  # or shutil.copy(source, destination)
