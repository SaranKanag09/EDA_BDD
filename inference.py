## Inference script for running YOLOv8 object detection on BDD100K validation dataset
# This script initializes a YOLOv8 model with pretrained weights, parses the BDD100K validation dataset,
# runs inference on each image, and saves the detection results in a JSON file formatted similarly to the original BDD100K labels.

import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

class BDD100KParser:
    def __init__(self, dataset_path, labels_file):
        """
        Initialize BDD100K dataset parser.
        
        Args:
            dataset_path: Path to BDD100K images directory
            labels_file: Path to BDD100K labels JSON file
        """
        self.dataset_path = Path(dataset_path)
        self.labels_file = Path(labels_file)
        self.labels = self._load_labels()
        print(f"Loaded {len(self.labels)} labels from {self.labels_file}")
    
    def _load_labels(self):
        """Load labels from JSON file."""
        with open(self.labels_file, 'r') as f:
            return json.load(f)
    
    def get_image_annotations(self, image_name):
        """Get annotations for a specific image."""
        for label in self.labels:
            if label['name'] == image_name:
                return label.get('labels', [])
        return []
    
    def parse_dataset(self):
        """Parse entire dataset and yield image paths with annotations."""
        image_files = list(self.dataset_path.glob('*.jpg'))
        
        for img_path in tqdm(image_files, desc="Parsing dataset"):
            annotations = self.get_image_annotations(img_path.name)
            yield str(img_path), annotations


class YOLOv8Detector:
    def __init__(self, model_name='yolov8m.pt'):
        """Initialize YOLOv8 detector with pretrained model."""
        self.model = YOLO(model_name)
    
    def detect(self, image_path, conf=0.5):
        """
        Run object detection on an image.
        
        Args:
            image_path: Path to input image
            conf: Confidence threshold
            
        Returns:
            Detection results
        """
        results = self.model.predict(image_path, conf=conf)
        return results[0]
    
    def visualize_detections(self, image_path, output_path, conf=0.5):
        """Detect and visualize results on image."""
        results = self.detect(image_path, conf)
        annotated_frame = results.plot()
        cv2.imwrite(output_path, annotated_frame)
    
    def extract_detections(self, results):
        """Extract detection details from results."""
        detections = []
        for box in results.boxes:
            detections.append({
                'class': self.model.names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': box.xyxy[0].tolist()
            })
        return detections


# Usage example
if __name__ == "__main__":
    
    DATASET_PATH = "./bdd100k_images_100k/bdd100k/images/100k/val"
    LABELS_FILE = "./bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
    MODEL_NAME = './checkpoints/yolov8m.pt'  # Path to pretrained YOLOv8 model
    OUTPUT_DIR = "./output"
    
    # Initialize parser
    parser = BDD100KParser(
        dataset_path=DATASET_PATH,
        labels_file=LABELS_FILE
    )
    
    # Initialize detector
    detector = YOLOv8Detector(model_name=MODEL_NAME)
    print("Detector initialized with YOLOv8m model.")
    
    # Process dataset
    # only for first 10 images for demonstration, remove the limit for full dataset
    for img_path, annotations in list(parser.parse_dataset())[:10]:
    #for img_path, annotations in parser.parse_dataset():
        results = detector.detect(img_path)
        detections = detector.extract_detections(results)        

    # Save detections in a JSON file similar to BDD100K format labels.json
    output_labels = []
    for img_path, annotations in list(parser.parse_dataset())[:10]:
        results = detector.detect(img_path)
        detections = detector.extract_detections(results)
        
        output_labels.append({
            'name': Path(img_path).name,
            'labels': detections
        })
    
    with open(f"{OUTPUT_DIR}/predictions.json", 'w') as f:
        json.dump(output_labels, f, indent=4)
    
    
    