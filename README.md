# Object detection on BDD100K dataset

This repo contains following tasks
- Exploratory Data Analysis(EDA) on BDD100K dataset
- Model inference on any pretrained models 
- Quantitative and Qualitative analysis between predictions and groundtruth
- Model training, validation and evaluation

## Setup 
1. Clone this repo.
2. Download BDD100K dataset and copy the folders *"bdd100K_images_100K"* and *"bdd100k_labels_release"* to cloned repo.
3. Create a python virtual environment and install packages mentioned in *"requirements.txt"*

## Exploratory Data Analysis(EDA) on BDD100K dataset

This repo contains creation of an interactive dashboard on BDD100K dataset. Please check folder EDA_dashboard for setup and details.
Also it contains EDA tab for detailed dataset details and EVAL tab for sample evaluation results. 

## Model inference on any pretrained models 

1. Make sure the preferred pretrained model is available in folder "./checkpoints/". In this example, yolov8m.pt is used. Download link: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m.pt
2. Verify the dataset_path, labels_file and model_name in *inference.py*. Below are considered in this example
   
    - DATASET_PATH ="bdd100k_images_100k/bdd100k/images/100k/val",
    - LABELS_FILE ="bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
    - MODEL_NAME = './checkpoints/yolov8m.pt'*
      
3. Run below command 
```
"python inference.py"
```
4. Output predictions will be saved in *"./output/predictions.json"*. Results are generated in COCO format as pretrained model was trained on COCO dataset. 

## Quantitative and Qualitative analysis between predictions and groundtruth

Predictions results generated in previous section are in COCO format, wheras the BDD100K dataset labels are in its original format. We can evaluate object detection metrics using metrics like 
- mAP : mean Average Precision
- mAR : mean Average Recall
- F1 score
These are most commonly used object detection metrics for quantitative analysis. Also one can visually plot the bounding boxes of predictions and ground truth to do a qualitative analysis. 

Now verify the paths for predictions, groundtruth and images inside eval.py. Below are the paths considered. 

- PREDS_JSON_PATH = "./output/predictions.json"
- GT_JSON_PATH = "./bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
- IMAGE_DIR = "./bdd100k_images_100k/bdd100k/images/100k/val" 
- OUTPUT_DIR = "./output/evaluation_results"

Once verified the file paths, run below command.
```
"python eval.py"
```
This generates a text file named *"evaluation_metrics.txt"* in *"./output/evaluation_results"*. Also generates 10 sample images with prediction and groundtruth labels in "Sample_results" folder. 

## Model training, validation and evaluation

In this section we can train a yolov8 model from scratch or retrain it for new dataset with some initial weights. This requires image and label pairs. 
In order to use yolov8, ultralytics package is used. This requires the labels to be in YOLO format. Existing BDD100k labels are in JSON files. 
Run below command to convert labels from BDD100k format to YOLO format. Also verify paths of JSON_DIR and OUTPUT_BASE before runnig script. 
```
python prepare_dataset_for_yolo.py
```
Once completed, the dataset folder will look as shown below <br />
 \bdd100k_images_100k\bdd100k\ <br />
  ├── images/ <br />
  │     ├── 100k/ <br />
  │     │    ├── train/  <-- (Contains .jpg files) <br />
  │     │    └── val/    <-- (Contains .jpg files) <br />
  │ <br />
  └── labels/            <-- (Create this folder if it's missing!) <br />
  │     ├── 100k/ <br />
  │     │    ├── train/  <-- (Contains .txt files) <br />
  │     │    └── val/    <-- (Contains .txt files) <br />

Now, run below command for initiating training on BDD100K images. Verify and modify training parameters as per need. 
This code is intially set to run a test run of 1 EPOCH. 
To run a smoke test on very small dataset(*fraction = 0.001 for 1% of training*), set flag IS_TEST_RUN = TRUE. 
```
python training.py
```
Ultralytics creates a folder named *"./runs/detect/"* to store all experiment details like weights, args and results. 
Once done checkpoints are saved in *"./checkpoints/"* folder. 
