import json
import os
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image, ImageDraw

st.set_page_config(page_title="BDD100K Dashboard", layout="wide")

st.title("🚗 BDD100K Interactive Dashboard")

# Create Top-Level Navigation Tabs
tab_eda, tab_eval = st.tabs(["📊 Dataset EDA", "🎯 Model Evaluation (EVAL)"])

# Internal container paths for both splits
PATHS = {
    "Train": {
        "json": "/app/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json",
        "images": "/app/data/bdd100k_images_100k/bdd100k/images/100k/train/"
    },
    "Validation": {
        "json": "/app/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json",
        "images": "/app/data/bdd100k_images_100k/bdd100k/images/100k/val/"
    }
}

# Evaluation Configuration Paths
EVAL_METRICS_PATH = "/app/output/evaluation_results/evaluation_metrics.txt"
EVAL_SAMPLES_DIR = "/app/output/evaluation_results/sample_results/"

# Optimized Intersection over Union (IoU) helper
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
        
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    return interArea / float(boxAArea + boxBArea - interArea)

# Global Cached Multi-Split Data Loader
@st.cache_data
def load_and_process_all_data(paths_dict):
    all_image_records = []
    all_object_records = []
    
    TINY_SIZE_THRESHOLD = 5
    IOU_THRESHOLD = 0.20
    
    for split_name, paths in paths_dict.items():
        json_path = paths["json"]
        if not os.path.exists(json_path):
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        for item in data:
            img_name = item.get("name")
            attrs = item.get("attributes", {})
            labels = item.get("labels", [])
            
            weather = attrs.get("weather", "undefined")
            scene = attrs.get("scene", "undefined")
            time_of_day = attrs.get("timeofday", "undefined")
            
            all_image_records.append({
                "image_name": img_name,
                "split": split_name,
                "weather": weather,
                "scene": scene,
                "time_of_day": time_of_day,
                "total_objects": len(labels)
            })
            
            frame_boxes = []
            for label in labels:
                category = label.get("category", "unknown")
                if category.strip() in ["driving area", "lane"]:
                    continue
                    
                box2d = label.get("box2d", {})
                if not box2d:
                    continue
                    
                xmin, ymin = box2d.get("x1", 0), box2d.get("y1", 0)
                xmax, ymax = box2d.get("x2", 0), box2d.get("y2", 0)
                w, h = xmax - xmin, ymax - ymin
                area = w * h
                
                aspect_ratio = float(w) / float(h) if h > 0 else 1.0
                is_undersized = (w < TINY_SIZE_THRESHOLD) or (h < TINY_SIZE_THRESHOLD)
                
                frame_boxes.append({
                    "category": category,
                    "coords": (xmin, ymin, xmax, ymax),
                    "w": w, "h": h, "area": area,
                    "aspect_ratio": aspect_ratio, "is_undersized": is_undersized
                })
            
            num_boxes = len(frame_boxes)
            for i in range(num_boxes):
                box1 = frame_boxes[i]
                is_overlapping = False
                
                for j in range(num_boxes):
                    if i != j:
                        if compute_iou(box1["coords"], frame_boxes[j]["coords"]) > IOU_THRESHOLD:
                            is_overlapping = True
                            break
                
                all_object_records.append({
                    "image_name": img_name,
                    "split": split_name,
                    "category": box1["category"],
                    "aspect_ratio": box1["aspect_ratio"],
                    "is_undersized": box1["is_undersized"],
                    "is_overlapping": is_overlapping,
                    "box_width": box1["w"],
                    "box_height": box1["h"],
                    "box_area": box1["area"],
                    "xmin": box1["coords"][0],
                    "ymin": box1["coords"][1],
                    "xmax": box1["coords"][2],
                    "ymax": box1["coords"][3],
                    "scene": scene 
                })
                
    return pd.DataFrame(all_image_records), pd.DataFrame(all_object_records)


# ==========================================
# 1. TAB 1: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
with tab_eda:
    with st.spinner("⏳ Parsing structural data profiles for Train and Validation sets..."):
        df_images, df_objects = load_and_process_all_data(PATHS)

    if df_images.empty:
        st.error("❌ Critical Error: Could not parse any dataset files. Verify your data directory volume mappings.")
    else:
        OBJECT_COLUMNS_TEMPLATE = df_objects.columns

        # --- Sidebar Controls (Scoped to EDA view via context explanation) ---
        st.sidebar.header("📂 Dataset Split Selector")
        split_options = ["Both Splits (Combined View)"] + list(PATHS.keys())
        selected_split = st.sidebar.selectbox("Choose Active Split Scope", options=split_options)

        st.sidebar.markdown("---")
        st.sidebar.header("🔍 Filter Parameters")
        all_classes = sorted(df_objects["category"].unique().tolist())
        selected_class = st.sidebar.selectbox("🎯 Target Object Class", options=["All Classes"] + all_classes)

        st.sidebar.markdown("---")
        st.sidebar.header("⚠️ Target Label Anomalies")
        anomaly_mode = st.sidebar.selectbox(
            "Isolate Dataset Anomalies",
            options=[
                "None (Show All)", 
                "Extreme Aspect Ratios (Top/Bottom 5%)", 
                "Tiny Boxes (< 5 Pixels)", 
                "Overlapping Boxes (IOU > 30%)", 
                "Missing labels"
            ]
        )

        # Storage Integrity Expansion
        with st.sidebar.expander("🛠️ Storage Integrity Diagnostics", expanded=False):
            if os.path.exists(PATHS["Train"]["images"]):
                physical_files = set(os.listdir(PATHS["Train"]["images"]))
                parsed_json_files = set(df_images[df_images["split"] == "Train"]["image_name"])
                missing_in_json = physical_files - parsed_json_files

                st.metric("Total Files on Disk", f"{len(physical_files):,}")
                st.metric("Total Parsed in JSON", f"{len(parsed_json_files):,}")
                st.metric("Image Files with no labels", f"{len(missing_in_json):,}")
            else:
                st.warning("Train image path mount is unreachable for integrity checks.")

        # Data filtering operations
        filtered_objects = df_objects.copy()
        filtered_images = df_images.copy()

        if selected_split != "Both Splits (Combined View)":
            filtered_objects = filtered_objects[filtered_objects["split"] == selected_split]
            filtered_images = filtered_images[filtered_images["split"] == selected_split]

        if selected_class != "All Classes":
            filtered_objects = filtered_objects[filtered_objects["category"] == selected_class]

        anomaly_alert = ""
        is_missing_labels_mode = False

        if anomaly_mode != "None (Show All)":
            if anomaly_mode == "Extreme Aspect Ratios (Top/Bottom 5%)":
                if not filtered_objects.empty:
                    lower_threshold = filtered_objects["aspect_ratio"].quantile(0.05)
                    upper_threshold = filtered_objects["aspect_ratio"].quantile(0.95)
                    filtered_objects = filtered_objects[(filtered_objects["aspect_ratio"] <= lower_threshold) | (filtered_objects["aspect_ratio"] >= upper_threshold)]
                    anomaly_alert = f"🚨 Displaying objects inside extreme aspect ratio bounds (Bottom 5%: ≤ {lower_threshold:.2f} or Top 5%: ≥ {upper_threshold:.2f})."
            elif anomaly_mode == "Tiny Boxes (< 5 Pixels)":
                filtered_objects = filtered_objects[filtered_objects["is_undersized"] == True]
                anomaly_alert = "🚨 Displaying objects smaller than 5 pixels."
            elif anomaly_mode == "Overlapping Boxes (IOU > 30%)":
                filtered_objects = filtered_objects[filtered_objects["is_overlapping"] == True]
                anomaly_alert = "🚨 Displaying objects sharing an IoU > 20% configuration."
            elif anomaly_mode == "Missing labels":
                is_missing_labels_mode = True
                anomaly_alert = "🚨 Displaying physical images containing no matching label entries."

        if is_missing_labels_mode:
            active_splits = ["Train", "Validation"] if selected_split == "Both Splits (Combined View)" else [selected_split]
            missing_records = []
            for split in active_splits:
                dir_path = PATHS[split]["images"]
                if os.path.exists(dir_path):
                    disk_files = set(os.listdir(dir_path))
                    labeled_files = set(df_images[df_images["split"] == split]["image_name"])
                    for file_name in (disk_files - labeled_files):
                        missing_records.append({"image_name": file_name, "split": split, "weather": "N/A", "scene": "N/A"})
            filtered_images = pd.DataFrame(missing_records) if missing_records else pd.DataFrame(columns=["image_name", "split", "weather", "scene"])
            filtered_objects = pd.DataFrame(columns=OBJECT_COLUMNS_TEMPLATE)
        else:
            target_image_names = set(filtered_objects["image_name"])
            filtered_images = filtered_images[filtered_images["image_name"].isin(target_image_names)]

        # Render Metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Total Filtered Images", f"{len(filtered_images):,}")
        m_col2.metric("Matching Label Instances", f"{len(filtered_objects):,}")
        m_col3.metric("Unique Classes Represented", filtered_objects["category"].nunique() if not filtered_objects.empty else 0)
        
        if anomaly_alert: st.warning(anomaly_alert)
        st.markdown("---")

        # Charts Panel
        if is_missing_labels_mode:
            st.info("📊 Chart representations are unavailable for unlabelled assets.")
        else:
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.subheader("⛅ Weather Based Distribution")
                if not filtered_images.empty:
                    fig_w = px.bar(filtered_images.groupby(["weather", "split"]).size().reset_index(name="count"), x="weather", y="count", color="split", barmode="group", text_auto=True, color_discrete_sequence=px.colors.qualitative.Safe)
                    st.plotly_chart(fig_w, use_container_width=True)
            with chart_col2:
                st.subheader("📈 Scene Based Distribution")
                if not filtered_objects.empty:
                    fig_s = px.bar(filtered_objects.groupby(["scene", "split"]).size().reset_index(name="count"), x="scene", y="count", color="split", barmode="group", color_discrete_sequence=px.colors.qualitative.Plotly)
                    st.plotly_chart(fig_s, use_container_width=True)

            st.subheader("📊 Class-Wise Distribution Breakdown")
            if not filtered_objects.empty:
                total_ordering = filtered_objects["category"].value_counts().index.tolist()
                fig_c = px.bar(filtered_objects.groupby(["category", "split"]).size().reset_index(name="count"), x="count", y="category", color="split", orientation='h', barmode="group", text_auto=True, category_orders={"category": total_ordering[::-1]}, color_discrete_sequence=px.colors.qualitative.Bold)
                fig_c.update_layout(height=max(400, len(total_ordering)*40))
                st.plotly_chart(fig_c, use_container_width=True)

        # Image Grid Explorer
        st.markdown("---")
        st.subheader("🖼️ Image Preview Gallery")
        active_gallery_split = "Train" if selected_split == "Both Splits (Combined View)" else selected_split
        IMAGES_DIR = PATHS[active_gallery_split]["images"]
        gallery_images = filtered_images[filtered_images["split"] == active_gallery_split] if not filtered_images.empty else pd.DataFrame()

        if not gallery_images.empty:
            gallery_items = gallery_images.head(10) if anomaly_mode != "None (Show All)" else gallery_images.iloc[0:8]
            grid_cols = st.columns(4)
            for idx, (_, row) in enumerate(gallery_items.iterrows()):
                col = grid_cols[idx % 4]
                path = os.path.join(IMAGES_DIR, row["image_name"])
                with col:
                    if os.path.exists(path):
                        img = Image.open(path).convert("RGB")
                        if anomaly_mode != "None (Show All)" and not is_missing_labels_mode:
                            draw = ImageDraw.Draw(img)
                            for _, box in filtered_objects[(filtered_objects["image_name"] == row["image_name"]) & (filtered_objects["split"] == active_gallery_split)].iterrows():
                                draw.rectangle([box["xmin"], box["ymin"], box["xmax"], box["ymax"]], outline="blue", width=3)
                        st.image(img, caption=row["image_name"], use_container_width=True)
                    else:
                        st.error(f"Missing asset: {row['image_name']}")
        else:
            st.info("No matching images found for this scope.")


# ==========================================
# 2. TAB 2: MODEL EVALUATION (EVAL)
# ==========================================
with tab_eval:
    st.header("🎯 Predictive Checkpoint Performance Analytics")
    
    # Text File Parsing Logic
    if os.path.exists(EVAL_METRICS_PATH):
        with open(EVAL_METRICS_PATH, "r") as f:
            metrics_raw_text = f.read()
            
        # Parse metrics dynamically into dictionary values for high-level cards
        metrics_dict = {}
        for line in metrics_raw_text.split("\n"):
            if ":" in line:
                key, val = line.split(":", 1)
                metrics_dict[key.strip()] = val.strip()

        # Display Top Configuration Subheaders
        dataset_meta = metrics_dict.get("Dataset", "BDD100K Validation Set")
        chkpt_meta = metrics_dict.get("Checkpoint", "Yolov8 Predictions vs BDD100K Ground Truth")
        st.markdown(f"**Dataset Target:** `{dataset_meta}` | **Evaluation Run Strategy:** `{chkpt_meta}`")
        
        # Row 1: Primary Mean Average Precision Metrics
        st.subheader("📈 Mean Average Precision (mAP) Benchmarks")
        ev_c1, ev_c2, ev_c3 = st.columns(3)
        ev_c1.metric("mAP (IoU 0.50:0.95)", metrics_dict.get("mAP (IoU 0.50:0.95)", "N/A"))
        ev_c2.metric("mAP @ 0.50 IoU", metrics_dict.get("mAP @ 0.50 IoU", "N/A"))
        ev_c3.metric("mAP @ 0.75 IoU", metrics_dict.get("mAP @ 0.75 IoU", "N/A"))
        
        # Row 2: Bounding Box Scale Specific Resolution
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🔍 Object Resolution Performance Scales")
        sz_c1, sz_c2, sz_c3 = st.columns(3)
        sz_c1.metric("mAP (Small Objects)", metrics_dict.get("mAP (Small objects)", "N/A"), help="Bounding box areas smaller than 32x32 pixels.")
        sz_c2.metric("mAP (Medium Objects)", metrics_dict.get("mAP (Medium objects)", "N/A"), help="Bounding box areas between 32x32 and 96x96 pixels.")
        sz_c3.metric("mAP (Large Objects)", metrics_dict.get("mAP (Large objects)", "N/A"), help="Bounding box dimensions exceeding 96x96 pixels.")
        
        # Row 3: Standard Machine Learning Confusion Statistics
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🎯 Precision, Recall & Confidence Summary Metrics")
        st_c1, st_c2, st_c3, st_c4 = st.columns(4)
        st_c1.metric("Precision", metrics_dict.get("Precision", "N/A"))
        st_c2.metric("Recall", metrics_dict.get("Recall", "N/A"))
        st_c3.metric("F1-Score", metrics_dict.get("F1-Score", "N/A"))
        st_c4.metric("MAR @ 100 max dets", metrics_dict.get("MAR @ 100 max dets", "N/A"))
        
        # Expandable Original File Terminal Logs View
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📄 View Raw evaluation_metrics.txt Output Logs", expanded=False):
            st.code(metrics_raw_text, language="text")
    else:
        st.error(f"❌ Evaluation metrics log missing at context path destination: `{EVAL_METRICS_PATH}`")

    # Image Verification Gallery Section
    st.markdown("---")
    st.subheader("🖼️ Prediction Sample Visual Gallery Output")
    
    if os.path.exists(EVAL_SAMPLES_DIR):
        # Scan for standard evaluation image file payloads
        sample_images = [f for f in os.listdir(EVAL_SAMPLES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if sample_images:
            # Enforce absolute ceiling limit profile constraints to target the requested 10 items
            selected_samples = sorted(sample_images)[:10]
            st.info(f"🎯 Displaying the first **{len(selected_samples)}** visualized model prediction image snapshots found inside directory `{EVAL_SAMPLES_DIR}`")
            
            # Draw Dynamic Evaluation Image Grid structure
            eval_grid_cols = st.columns(2)  # Larger 2-column layout for detailed prediction inspection
            for val_idx, image_filename in enumerate(selected_samples):
                grid_target_col = eval_grid_cols[val_idx % 2]
                full_sample_img_path = os.path.join(EVAL_SAMPLES_DIR, image_filename)
                
                with grid_target_col:
                    try:
                        loaded_sample_img = Image.open(full_sample_img_path)
                        st.image(loaded_sample_img, caption=f"Prediction Sample: {image_filename}", use_container_width=True)
                    except Exception as img_err:
                        st.error(f"Failed to render workspace asset {image_filename}: {img_err}")
        else:
            st.warning(f"⚠️ No image target files found inside evaluation folder destination: `{EVAL_SAMPLES_DIR}`")
    else:
        st.error(f"❌ Target prediction samples directory workspace folder path was unreachable: `{EVAL_SAMPLES_DIR}`")