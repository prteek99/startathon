import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# --- CONFIGURATION ---
EXTRACT_PATH = r"F:\hackathon\Dataset_Extracted"  # Path to your extracted dataset
MODEL_PATH = r"F:\hackathon\best_multiclass_model.pth" # Path to your trained model weights
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (320, 320)
NUM_CLASSES = 10
SAMPLE_SIZE = 50  # Number of random images to test for IoU calculation

# --- COLOR MAP ---
# [Red, Green, Blue] - Designed for high contrast visualization
COLOR_MAP = np.array([
    [0, 128, 0],    # 0: Trees (Dark Green)
    [0, 255, 0],    # 1: Lush Bushes (Bright Green)
    [200, 200, 0],  # 2: Dry Grass (Yellow)
    [100, 100, 0],  # 3: Dry Bushes (Olive)
    [64, 0, 0],     # 4: Ground Clutter (Dark Red)
    [255, 0, 255],  # 5: Flowers (Pink)
    [139, 69, 19],  # 6: Logs (Brown)
    [128, 128, 128],# 7: Rocks (Gray)
    [255, 0, 0],    # 8: Landscape (Red)
    [0, 0, 255]     # 9: Sky (Blue)
], dtype=np.uint8)

# Class Names for Report
CLASS_NAMES = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter", 
    "Flowers", "Logs", "Rocks", "Landscape", "Sky"
]

# Mapping Raw IDs (e.g., 100) to Model IDs (e.g., 0)
CLASS_MAPPING = {100:0, 200:1, 300:2, 500:3, 550:4, 600:5, 700:6, 800:7, 7100:8, 10000:9}

# --- HELPER FUNCTIONS ---
def encode_mask(mask):
    """
    Converts a raw mask (with IDs like 100, 200) into a model-compatible mask (0, 1, 2...).
    Also ensures the mask is strictly 2D (Grayscale).
    """
    if len(mask.shape) == 3:
        mask = mask[:, :, 0] # Force 2D if loaded as 3D
        
    mask_mapped = np.full(mask.shape, 8, dtype=np.longlong) # Default to Landscape (Class 8)
    for raw_id, map_id in CLASS_MAPPING.items():
        mask_mapped[mask == raw_id] = map_id
    return mask_mapped

def colorize_mask(mask):
    """Converts a segment map (0-9) to an RGB image for visualization."""
    return COLOR_MAP[mask]

def calculate_iou(pred_mask, true_mask):
    """Calculates Intersection over Union (IoU) for each class."""
    iou_list = []
    for cls in range(NUM_CLASSES):
        pred_inds = pred_mask == cls
        target_inds = true_mask == cls
        
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        
        if union == 0:
            iou_list.append(float('nan')) # Ignore classes not present in the image
        else:
            iou_list.append(intersection / union)
    return iou_list

def find_data_folders():
    """Automatically locates the image and mask directories."""
    img_dir, mask_dir = None, None
    
    # Priority: Validation Folder
    for root, dirs, files in os.walk(EXTRACT_PATH):
        if "val" in root.lower() and "Color_Images" in os.path.basename(root): img_dir = root
        if "val" in root.lower() and ("Segmentation" in os.path.basename(root) or "Label" in os.path.basename(root)): mask_dir = root

    # Fallback: Training Folder (if Val empty)
    if not img_dir or not mask_dir:
        for root, dirs, files in os.walk(EXTRACT_PATH):
            if "train" in root.lower() and "Color_Images" in os.path.basename(root): img_dir = root
            if "train" in root.lower() and ("Segmentation" in os.path.basename(root) or "Label" in os.path.basename(root)): mask_dir = root
            
    return img_dir, mask_dir

# --- MAIN TEST FUNCTION ---
def run_test():
    print("🚀 Duality AI Hackathon - Final Evaluation Script")
    print("="*50)

    # 1. Setup Data
    img_dir, mask_dir = find_data_folders()
    if not img_dir or not mask_dir:
        print("❌ Error: Could not locate dataset folders. Check EXTRACT_PATH.")
        return
    print(f"📂 Images: {img_dir}")
    print(f"📂 Masks:  {mask_dir}")

    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model weights not found at {MODEL_PATH}")
        return

    print(f"⏳ Loading Model from {os.path.basename(MODEL_PATH)}...")
    model = smp.UnetPlusPlus(
        encoder_name="resnet34", 
        encoder_weights=None, 
        in_channels=3, 
        classes=NUM_CLASSES
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 3. Select Random Samples
    all_images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    if len(all_images) > SAMPLE_SIZE:
        print(f"🎲 Randomly selecting {SAMPLE_SIZE} images for evaluation...")
        selected_images = random.sample(all_images, SAMPLE_SIZE)
    else:
        selected_images = all_images

    # 4. Processing Loop
    total_ious = []
    processed_samples = [] # To store images for visualization

    print("⚡ Running Inference...")
    transform = A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1], interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    for img_name in tqdm(selected_images):
        img_path = os.path.join(img_dir, img_name)
        
        # Robust Mask Finding Logic
        candidates = [img_name, img_name.replace(".jpg", ".png"), "Label_" + img_name.replace(".jpg", ".png")]
        mask_path = None
        for cand in candidates:
            if os.path.exists(os.path.join(mask_dir, cand)):
                mask_path = os.path.join(mask_dir, cand)
                break
        
        if not mask_path: continue

        # Load Data
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, -1) # Read Raw (16-bit support)
        mask = encode_mask(mask) # Handle IDs and 2D conversion
        
        # Preprocess
        aug = transform(image=image, mask=mask)
        img_tensor = aug["image"].unsqueeze(0).to(DEVICE)
        true_mask = aug["mask"].numpy()

        # Inference
        with torch.no_grad():
            logits = model(img_tensor)
            pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

        # Calculate Metric
        ious = calculate_iou(pred_mask, true_mask)
        total_ious.append(ious)

        # Store for visualization
        processed_samples.append({
            "image": cv2.resize(image, (IMG_SIZE[1], IMG_SIZE[0])),
            "true": cv2.resize(mask, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_NEAREST),
            "pred": pred_mask
        })

    # 5. Report Generation
    if not total_ious:
        print("❌ No images were processed successfully.")
        return

    total_ious = np.array(total_ious)
    mean_ious = np.nanmean(total_ious, axis=0)
    final_score = np.nanmean(mean_ious)

    print("\n📊 EVALUATION REPORT")
    print("="*40)
    print(f"{'CLASS NAME':<20} | {'IoU SCORE':<10}")
    print("-" * 40)
    for i, score in enumerate(mean_ious):
        print(f"{CLASS_NAMES[i]:<20} | {score:.4f}")
    print("="*40)
    print(f"🏆 FINAL MEAN IoU: {final_score:.4f}")
    print("="*40)

    # 6. Visualization
    print("🖼️ Displaying Random Results...")
    if len(processed_samples) > 3:
        visual_samples = random.sample(processed_samples, 3)
    else:
        visual_samples = processed_samples

    plt.figure(figsize=(15, 10))
    for i, sample in enumerate(visual_samples):
        plt.subplot(3, 3, i*3 + 1)
        plt.imshow(sample["image"])
        plt.title("Original")
        plt.axis("off")

        plt.subplot(3, 3, i*3 + 2)
        plt.imshow(colorize_mask(sample["true"]))
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(3, 3, i*3 + 3)
        plt.imshow(colorize_mask(sample["pred"]))
        plt.title("Prediction")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_test()