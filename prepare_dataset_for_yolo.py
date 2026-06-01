# This script converts BDD100K labels from their original JSON format into the YOLOv8 format 
# (text files with normalized bounding box coordinates). It reads the combined JSON files for 
# training and validation, extracts the relevant bounding box information for the specified classes, 
# and saves them in the appropriate directory structure for YOLOv8 training.

import json
import os
from pathlib import Path
from tqdm import tqdm

def convert_bdd100k_to_yolo(json_path, output_label_dir, img_width=1280, img_height=720):
    """
    Converts a BDD100K labels JSON file into individual YOLOv8 text files.
    Default BDD100K resolution is 1280x720.
    """
    # Mapping BDD100K classes to YOLO dataset configuration IDs
    class_map = {
        'pedestrian': 0, 'rider': 1, 'car': 2, 'truck': 3, 'bus': 4,
        'train': 5, 'motorcycle': 6, 'bicycle': 7, 'traffic light': 8, 'traffic sign': 9
    }
    
    output_path = Path(output_label_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        labels_data = json.load(f)
        
    print("Converting labels to YOLO format...")
    for item in tqdm(labels_data):
        image_name = item['name']
        # Remove extension (.jpg) to get the base text file name
        label_file_name = Path(image_name).stem + '.txt'
        label_file_path = output_path / label_file_name
        
        # Open a text file for this specific image
        with open(label_file_path, 'w') as out_f:
            if 'labels' not in item:
                continue
                
            for label in item['labels']:
                category = label['category']
                
                # Check if category is in our target tracking list
                if category not in class_map:
                    continue
                    
                # Ensure it has a bounding box (ignoring polylines/lane markings here)
                if 'box2d' not in label:
                    continue
                    
                box = label['box2d']
                x1 = box['x1']
                y1 = box['y1']
                x2 = box['x2']
                y2 = box['y2']
                
                # Convert absolute coordinates to YOLO normalized coordinates
                # YOLO format: x_center, y_center, width, height (all 0.0 to 1.0)
                box_width = x2 - x1
                box_height = y2 - y1
                x_center = x1 + (box_width / 2.0)
                y_center = y1 + (box_height / 2.0)
                
                # Normalize by image dimensions
                x_center /= img_width
                y_center /= img_height
                box_width /= img_width
                box_height /= img_height
                
                class_id = class_map[category]
                
                # Write to the file
                out_f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

# --- EXECUTION ---
if __name__ == '__main__':
    
    # Existing paths to where bdd100K combined JSON files are located. 
    JSON_DIR = './bdd100k_labels_release/bdd100k/labels'
    
    # Target label directories where labels will be saved in YOLO format in text files 
    OUTPUT_BASE = './bdd100k_images_release/bdd100k/labels/100k'  
    
    # Process Train JSON
    train_json = os.path.join(JSON_DIR, 'bdd100k_labels_images_train.json') 
    if os.path.exists(train_json):
        convert_bdd100k_to_yolo(train_json, os.path.join(OUTPUT_BASE, 'train'))
        
    # Process Val JSON
    val_json = os.path.join(JSON_DIR, 'bdd100k_labels_images_val.json')
    if os.path.exists(val_json):
        convert_bdd100k_to_yolo(val_json, os.path.join(OUTPUT_BASE, 'val'))

    print("Conversion complete!")