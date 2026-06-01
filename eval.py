import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# =====================================================================
# CONFIGURATION to set file paths 
# =====================================================================
PREDS_JSON_PATH = "./output/predictions.json"
GT_JSON_PATH = "./bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
IMAGE_DIR = "./bdd100k_images_100k/bdd100k/images/100k/val"          
OUTPUT_DIR = "./output/evaluation_results"

# Define all possible classes in dataset mapped to unique integers
LABEL_MAP = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "motorcycle": 3,
    "pedestrian": 4,
    "rider": 5,
    "train": 6,
    "traffic light": 7,
    "traffic sign": 8,
    "bicycle": 9
}
# =====================================================================

def prepare_predictions(preds_json, label_to_idx):
    """Parses prediction format and converts it to the expected format for evaluation."""
    formatted_data = {}
    for item in preds_json:
        img_name = item["name"]
        boxes, labels, scores = [], [], []
        
        for annotation in item.get("labels", []):
            class_name = annotation["class"]
            if class_name in label_to_idx:
                boxes.append(annotation["bbox"])
                labels.append(label_to_idx[class_name])
                scores.append(annotation.get("confidence", 1.0))
                
        formatted_data[img_name] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64),
            "scores": torch.tensor(scores, dtype=torch.float32) if scores else torch.empty((0,), dtype=torch.float32)
        }
    return formatted_data


def prepare_ground_truth_bdd100k(gt_json, label_to_idx):
    """Parses BDD100K Ground Truth format and converts it to the expected format for evaluation."""
    formatted_data = {}
    for item in gt_json:
        img_name = item["name"]
        boxes, labels = [], []
        
        for annotation in item.get("labels", []):
            class_name = annotation.get("category")
            if class_name in label_to_idx and "box2d" in annotation:
                box2d = annotation["box2d"]
                # Convert dict to expected flat list: [xmin, ymin, xmax, ymax]
                bbox = [box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]]
                
                boxes.append(bbox)
                labels.append(label_to_idx[class_name])
                
        formatted_data[img_name] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
        }
    return formatted_data
 
def export_metrics_report(results, output_path):
    """Saves evaluation summary metrics safely to a text file without version crashes."""
    
    # Helper to safely extract tensor items if the key exists
    def get_map_val(key):
        if key in results:
            return f"{results[key].item():.4f}"
        return "N/A (Not computed)"
        
    # 2. Derive F1-Score from the extracted precision and recall
    precision = results["map_50"]  # Using AP@50 as a proxy for precision
    recall = results["mar_100"]     # Using AR@100 as a proxy for recall

    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    print(f"Derived F1-Score: {f1_score.item()}")

    report_lines = [
        "================ EVALUATION METRICS ================",
        "Dataset: BDD100K Validation Set",
        "Checkpoint: Yolov8 Predictions vs BDD100K Ground Truth",
        f"mAP (IoU 0.50:0.95) : {get_map_val('map')}",
        f"mAP @ 0.50 IoU      : {get_map_val('map_50')}",
        f"mAP @ 0.75 IoU      : {get_map_val('map_75')}",
        f"mAP (Small objects) : {get_map_val('map_small')}",
        f"mAP (Medium objects): {get_map_val('map_medium')}",
        f"mAP (Large objects) : {get_map_val('map_large')}",
        "----------------------------------------------------",
        f"MAR @ 100 max dets  : {get_map_val('mar_100') if 'mar_100' in results else get_map_val('mar_max_100')}",
        f"Precision           : {precision:.4f}",
        f"Recall              : {recall:.4f}",
        f"F1-Score            : {f1_score.item():.4f}",
        "===================================================="
    ]
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    with open(output_path, "w") as f:
        f.write(report_text)
    print(f"\n[INFO] Saved metric summary report to: {output_path}")

def export_visualizations(preds_dict, gts_dict, image_dir, output_dir, num_samples=10):
    """Generates and saves side-by-side comparison figures for the first N samples."""
    common_images = list(set(preds_dict.keys()).intersection(set(gts_dict.keys())))[:num_samples]
    idx_to_label = {v: k for k, v in LABEL_MAP.items()}
    
    if not common_images:
        print("[WARNING] No matching image names found between datasets to plot.")
        return

    print(f"[INFO] Exporting {len(common_images)} sample visualizations to '{output_dir}'...")
    
    for idx, img_name in enumerate(common_images):
        pred_data = preds_dict[img_name]
        gt_data = gts_dict[img_name]
        
        img_path = os.path.join(image_dir, img_name) if image_dir else ""
        if image_dir and os.path.exists(img_path):
            img_gt = Image.open(img_path)
            img_pred = img_gt.copy()
            width, height = img_gt.size
        else:
            all_boxes = torch.cat([pred_data["boxes"], gt_data["boxes"]], dim=0)
            width = int(all_boxes[:, 2].max().item() + 50) if len(all_boxes) > 0 else 1280
            height = int(all_boxes[:, 3].max().item() + 50) if len(all_boxes) > 0 else 720
            img_gt, img_pred = None, None

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # --- Ground Truth Panel (Left) ---
        ax_gt = axes[0]
        ax_gt.set_title(f"GT (BDD100K Format): {img_name}", fontsize=12, weight='bold')
        if img_gt:
            ax_gt.imshow(img_gt)
        else:
            ax_gt.set_xlim(0, width); ax_gt.set_ylim(height, 0); ax_gt.set_aspect('equal')
            
        for box, label_idx in zip(gt_data["boxes"], gt_data["labels"]):
            xmin, ymin, xmax, ymax = box.tolist()
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                     linewidth=2, edgecolor='lime', facecolor='none')
            ax_gt.add_patch(rect)
            ax_gt.text(xmin, ymin - 5, idx_to_label[label_idx.item()], 
                       color='lime', fontsize=9, weight='bold', 
                       bbox=dict(facecolor='black', alpha=0.6, pad=1, edgecolor='none'))

        # --- Prediction Panel (Right) ---
        ax_pred = axes[1]
        ax_pred.set_title(f"Predictions: {img_name}", fontsize=12, weight='bold')
        if img_pred:
            ax_pred.imshow(img_pred)
        else:
            ax_pred.set_xlim(0, width); ax_pred.set_ylim(height, 0); ax_pred.set_aspect('equal')
            
        for box, label_idx, score in zip(pred_data["boxes"], pred_data["labels"], pred_data.get("scores", [])):
            if score.item() < 0.25: # Confidence filtering
                continue
            xmin, ymin, xmax, ymax = box.tolist()
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax_pred.add_patch(rect)
            ax_pred.text(xmin, ymin - 5, f"{idx_to_label[label_idx.item()]} {score.item():.2f}", 
                       color='red', fontsize=9, weight='bold', 
                       bbox=dict(facecolor='black', alpha=0.6, pad=1, edgecolor='none'))

        # Save plot
        clean_name = os.path.splitext(img_name)[0]
        out_img_path = os.path.join(output_dir, f"sample_results/sample_{idx+1}_{clean_name}_eval.png")
        plt.tight_layout()
        plt.savefig(out_img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"[INFO] Reading structured log paths...")
    with open(PREDS_JSON_PATH, "r") as f:
        preds_json = json.load(f)
    with open(GT_JSON_PATH, "r") as f:
        gts_json = json.load(f)
        
    print("[INFO] Transforming inputs into calculation tensors...")
    preds_parsed = prepare_predictions(preds_json, LABEL_MAP)
    gts_parsed = prepare_ground_truth_bdd100k(gts_json, LABEL_MAP)
    
    # Align datasets by keys
    common_keys = [k for k in preds_parsed.keys() if k in gts_parsed]
    preds_list = [preds_parsed[k] for k in common_keys]
    gts_list = [gts_parsed[k] for k in common_keys]
    
    print("[INFO] Running torchmetrics evaluation calculation engine...")
    metric_engine = MeanAveragePrecision(box_format='xyxy', class_metrics=False)
    metric_engine.update(preds_list, gts_list)
    results = metric_engine.compute()    
    
    # Save files
    export_metrics_report(results, os.path.join(OUTPUT_DIR, "evaluation_metrics.txt"))
    export_visualizations(preds_parsed, gts_parsed, IMAGE_DIR, OUTPUT_DIR, num_samples=10)
    
    print("\n[SUCCESS] Execution finished safely. Check the 'output' directory.")