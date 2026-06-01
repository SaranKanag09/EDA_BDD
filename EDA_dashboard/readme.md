# BDD100K exploratory dataset analysis 
This dataset is an object detection dataset containing 12 classes. It is mostly used for training detection models for autonomous driving problem statement. This data is diversly spread across variations in terms of scene, weather and climatic conditions. A huge chunk of 70K images for training, 10K for validation and 20K for testing are considered. 

This interactive dashboard contains two tabs. EDA tab discussed in section 1 and EVAL tab discussed in section 2.

## Section 1: EDA tab 

1. **Storage Integrity Diagnostics**: In this section, we can see the total amount of images and labels read and processed. One can also check if all images contain labels. This dataset have 137 samples without annotations. 
2. **Class wise distribution of train and validation(val) sets**: This provides a clear spread of considered 10/12 classes over train and val set. This helps to identify class imbalance.
3. **Weather wise distribution of train and val sets**: This helps to identify the spread of data over different climatic conditions like cloudy, snow, foggy or different timings of event like daytime, night. 
4. **Scene wise distribution of train and val sets**: This helps in understanding the distribution of data across various scenes like city street, gas station, highway, parking lot, etc,.  
5. **Detailed anomalies identification within dataset based on below categories**   
    a. *Aspect Ratio of bounding box:-* Top 5% higher and top 5% lower percentile of aspect ratio. This helps to identify labels which are drawn too long or too wide which sometimes can be an anomaly. <br>
    b. *Tiny bounding box:-*  Which are having width or height of bounding box lesser than 5 pixels. When we visualize these samples, these can be mostly for far objects. There are high chances this labels can be incorrect.<br> 
    c. *Overlapping bounding box:-* In most of cases we may have several bounding boxes overlapping one over other or one adjacent to other. This can be identified by computing their IoU. An IoU overlap of 30% is considered to visually inspect. <br>
    d. *Missing labels:-* In given dataset though there are enough samples, there can be images without labels. In this set considered, there were few samples with no labels. <br>
7. **Image gallery**: To visualize sample images on the fly as we filter id provided in this dashboard. Also one can visualize BBOX drawn over filtered samples when playing around with anomalies. 

## Section 2: EVAL tab

This section mainly focuses on displaying outcome of evaluation results performed between predictions and groundtruth. For experimental purpose, we consider predictions from "EDA_BDD/output/predictions.json".
Groundtruth labels are considered from the dataset downloaded. 
This enables quantitative and qualitative analysis over results. 

### Steps to launch this interactive dashboard

**Prerequisites** 

1. Install Docker desktop (in windows).
2. Install VS code and Docker extension in VS code. Note:- Any equivalent can be used to setup and build docker containers.
3. Download BDD100K dataset and have it in your local directory. Below is the expected folder structure.

.\bdd100k_images_100k\bdd100k\ <br />
  ├── images/ <br />
  │     ├── 100k/ <br />
  │     │    ├── train/  <-- (Contains .jpg files) <br />
  │     │    └── val/    <-- (Contains .jpg files) <br />
  │ <br />
.\bdd100k_labels_release\bdd100k\  <-- (Create this folder if it's missing!)  <br />
  │     └── labels/ <br />
  │         ├── bdd100k_labels_images_train.json/  <-- (generated .txt files here) <br />
  │         └── bdd100k_labels_images_val.json/    <-- (generated .txt files here) <br />


4. This repo also inputs *predictions.json* and "sample_results" folder from *EDA_BDD/output/*.

**Dashboard launch using self contained docker**

1. Clone this repo
2. Change directory to one containing Dockerfile "EDA_BDD/EDA_dashboard". 
3. Build docker using below command
```
docker build -t <docker_name> .
```
4. Running the docker container. Once the docker build is successful. Run docker container. Before running docker container replace absolute path of dataset with your dataset path. 
```
docker run -p 8501:8501 -v ./bdd100k_labels_release/bdd100k/labels/:/app/data/bdd100k_labels_release/bdd100k/labels/:ro -v ./bdd100k_images_100k/bdd100k/images/100k/:/app/data/bdd100k_images_100k/bdd100k/images/100k/:ro -v ./EDA_BDD/output/:/app/output/:ro <docker_name>
```
5. Open dashboard. Once the docker run command runs successfully. Open a new browser and enter *https://localhost:8501*. This launches the dashboard with all details we made.
6. Wait for some time until the data processing happens completely. Once done, play around with options given to experience the EDA and EVAL. 
