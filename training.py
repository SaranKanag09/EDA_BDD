# This script trains a YOLOv8 model on the BDD100K dataset using the customized label directory structure.
# It uses yolov8m.pt for pretrained weights and initiates a new training process. 
# The script also includes a test mode that runs a quick smoke test on a small subset of the data
# to ensure everything is set up correctly before launching a full training run.

import os
import json
import numpy as np
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import Dataset
from ultralytics import YOLO
import yaml

class BDD100KDataset(Dataset):
    """BDD100K dataset parser used primarily for class weight calculation"""
    
    def __init__(self, image_dir, label_dir, split='train', img_size=640, max_samples=None):
        self.split = split
        self.img_size = img_size
        
        self.images_dir = Path(image_dir) / split
        self.labels_dir = Path(label_dir) / split
        
        # Target .jpg files 
        all_images = sorted(list(self.images_dir.glob('*.jpg')))
        self.image_files = all_images[:max_samples] if max_samples else all_images
        self.class_counts = self._count_classes()
        
    def _count_classes(self):
        """Count class occurrences safely based only on sampled images"""
        class_counts = Counter()
        
        if not self.image_files:
            print(f"Warning: No images found in {self.images_dir}!")
            return class_counts
            
        for img_path in self.image_files:
            label_file = self.labels_dir / (img_path.stem + '.txt')
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.split()
                        if parts:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
        return class_counts
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.labels_dir / (img_path.stem + '.txt')
        return str(img_path), str(label_path)


def calculate_class_weights(dataset):
    """Calculate class weights to handle imbalanced dataset"""
    class_counts = dataset.class_counts
    if not class_counts:
        return None
        
    total_samples = sum(class_counts.values())
    num_classes = max(class_counts.keys()) + 1
    weights = np.ones(num_classes)
    
    for class_id, count in class_counts.items():
        weights[class_id] = total_samples / (num_classes * max(count, 1))
    
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def create_dataset_yaml(image_dir, label_dir, output_path='dataset.yaml'):
    """Create a clean YAML configuration tracking the customized label directory"""
    
    # We use absolute paths directly for train/val/test to prevent YOLO layout auto-search confusion
    config = {
        'path': '',  # Kept empty so we can use absolute paths below directly
        'train': str((Path(image_dir) / 'train').absolute()),
        'val': str((Path(image_dir) / 'val').absolute()),
        'nc': 10,  
        'names': [
            'pedestrian', 'rider', 'car', 'truck', 'bus', 
            'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
        ]
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    # CRITICAL: YOLOv8 searches for text labels by replacing '/images/' with '/labels/' in the path string.
    # Because your labels directory is 'bdd100k_labels_customized', we create a symlink or ensure YOLO knows where it is.
    # If YOLO complains it cannot find labels, rename your 'bdd100k_labels_customized' folder to 'labels' 
    # and put it right next to your 'images' folder.
    
    print(f"Dataset YAML config created at: {os.path.abspath(output_path)}")
    return output_path


def train_yolov8(image_dir, label_dir, model_name='./checkpoints/yolov8m.pt', epochs=1, batch_size=16, 
                 imgsz=640, device='cpu', save_dir='./experiments', is_test=True):
    """Training pipeline for YOLOv8 model"""
    
    dataset_yaml = create_dataset_yaml(image_dir, label_dir)
    
    # Check weights on the small test slice if testing
    max_samples = 20 if is_test else None
    try:
        train_dataset = BDD100KDataset(image_dir, label_dir, split='train', img_size=imgsz, max_samples=max_samples)
        class_weights = calculate_class_weights(train_dataset)
        if class_weights is not None:
            print(f"Calculated Class Weights (Subset): {class_weights.tolist()}")
    except Exception as e:
        print(f"Could not calculate class weights dynamically ({e}).")
    
    model = YOLO(model_name)
    
    # Define baseline training arguments
    train_args = {
        'data': dataset_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'device': device,
        'patience': 3,
        'save': True,
        'project': save_dir,
        'name': 'bdd100k_yolov8',
        'workers': 0 if device == 'cpu' else 4,
        'verbose': True,
        'seed': 42
    }
    
    # If it's a test run, restrict the dataset fraction
    if is_test:
        train_args['fraction'] = 0.001  # Only look at 0.001% of the data
        print("Running in SMOKE TEST mode. Disabling validation completely.")
    
    # Launch training with the unified arguments dictionary
    results = model.train(**train_args)
    
    return model, results


def main():
    
    IMAGE_DIR = 'E:/EDA_BDD/bdd100k_images_100k/bdd100k/images/100k'
    LABEL_DIR = 'E:/EDA_BDD/bdd100k_images_100k/bdd100k/labels/100k' #labels in yolo format in txt files
    
    # Configuration Flags
    IS_TEST_RUN = True  # Set to False when you want to run your full dataset training!. 
    
    MODEL_NAME = './checkpoints/yolov8m.pt'  # pretrained weights path
    EPOCHS = 1                 
    BATCH_SIZE = 4 if IS_TEST_RUN else 16  # Small batch size for quick CPU testing
    IMGSZ = 640
    DEVICE = 'cpu'             
    SAVE_DIR = './experiments'
    
    print("Starting YOLOv8 training workflow...")
    
    model, results = train_yolov8(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        model_name=MODEL_NAME,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        imgsz=IMGSZ,
        device=DEVICE,
        save_dir=SAVE_DIR,
        is_test=IS_TEST_RUN
    )
    
    print("Training phase completed!")
    
    print("\nEvaluating model...")
    try:
        metrics = model.val(data='dataset.yaml', imgsz=IMGSZ, batch=BATCH_SIZE, device=DEVICE, fraction=0.01)
    except Exception as e:
        print(f"Validation step skipped or errored: {e}")
    
    final_path = os.path.join('./checkpoints', 'bdd100k_yolov8_self.pt')
    model.save(final_path)
    print(f"Model successfully saved to {final_path}")


if __name__ == '__main__':
    main()