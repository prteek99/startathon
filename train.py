import os
import zipfile
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from tqdm import tqdm
import warnings

# --- CONFIGURATION ---
ZIP_PATH = r"F:\hackathon\Offroad_Segmentation_Training_Dataset.zip"
EXTRACT_PATH = r"F:\hackathon\Dataset_Extracted"
MODEL_SAVE_PATH = r"F:\hackathon\best_multiclass_model.pth"

# Hackathon PDF ke hisaab se IDs
# ID -> Model Index
CLASS_MAPPING = {
    100: 0,   # Trees
    200: 1,   # Lush Bushes
    300: 2,   # Dry Grass
    500: 3,   # Dry Bushes
    550: 4,   # Ground Clutter
    600: 5,   # Flowers
    700: 6,   # Logs
    800: 7,   # Rocks
    7100: 8,  # Landscape (Background)
    10000: 9  # Sky
}
NUM_CLASSES = len(CLASS_MAPPING) # Total 10 classes

EPOCHS = 50
BATCH_SIZE = 2  # RTX 2050 ke liye 2 safe hai (Heavy model)
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (320, 320)

warnings.filterwarnings("ignore")

# --- STEP 1: FOLDER DETECTION ---
def find_data_folders(root_path):
    print(f"🔍 Scanning {root_path}...")
    img_dir, mask_dir = None, None
    
    # Auto-detect logic
    for root, dirs, files in os.walk(root_path):
        if "train" in root.lower() and "val" not in root.lower():
            folder_name = os.path.basename(root)
            # Images dhoondo
            if "Color_Images" in folder_name or "images" in folder_name.lower():
                if len([f for f in files if f.endswith('.jpg')]) > 10:
                    img_dir = root
            # Masks dhoondo
            if "Segmentation" in folder_name or "Label" in folder_name or "masks" in folder_name.lower():
                if len([f for f in files if f.endswith('.png')]) > 10:
                    mask_dir = root

    if not img_dir or not mask_dir:
        # Hardcoded fallback for known Duality structure
        possible_img = os.path.join(root_path, "Offroad_Segmentation_Training_Dataset", "train", "Color_Images")
        possible_msk = os.path.join(root_path, "Offroad_Segmentation_Training_Dataset", "train", "Label_Images") # Or Segmentation
        if os.path.exists(possible_img): img_dir = possible_img
        # Try finding segmentation folder
        for root, dirs, files in os.walk(root_path):
            if "Segmentation" in root and "train" in root: mask_dir = root

    if not img_dir or not mask_dir:
        print(f"❌ ERROR: Folders nahi mile! \nScanned: {root_path}")
        exit()

    print(f"✅ Images Folder: {img_dir}")
    print(f"✅ Masks Folder:  {mask_dir}")
    return img_dir, mask_dir

# --- STEP 2: DATASET CLASS WITH MAPPING ---
class OffroadMulticlassDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.img_dir, self.mask_dir = find_data_folders(root_dir)
        self.pairs = []
        
        # Files list
        all_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        print("🔄 Matching pairs...")
        for filename in all_files:
            # Try specific mask naming patterns
            candidates = [
                filename, 
                filename.replace(".jpg", ".png"),
                "Label_" + filename.replace(".jpg", ".png")
            ]
            
            for cand in candidates:
                mask_path = os.path.join(self.mask_dir, cand)
                if os.path.exists(mask_path):
                    self.pairs.append((os.path.join(self.img_dir, filename), mask_path))
                    break
        
        print(f"✅ Total Pairs: {len(self.pairs)}")
        
        # Debug: Check First Mask Values
        if len(self.pairs) > 0:
            test_mask = cv2.imread(self.pairs[0][1], -1) # Read as is (16-bit support)
            unique_vals = np.unique(test_mask)
            print(f"🧐 DEBUG: First mask unique values found: {unique_vals}")
            print(f"👉 Expected IDs: {list(CLASS_MAPPING.keys())}")

    def __len__(self): return len(self.pairs)

    def encode_mask(self, mask):
        # Create an empty mask with label 8 (Landscape/Background) as default
        mask_mapped = np.full(mask.shape, 8, dtype=np.longlong) 
        
        for raw_id, map_id in CLASS_MAPPING.items():
            mask_mapped[mask == raw_id] = map_id
            
        return mask_mapped

    def __getitem__(self, index):
        img_path, mask_path = self.pairs[index]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Mask with -1 to support 16-bit IDs (like 10000)
        mask = cv2.imread(mask_path, -1) 
        
        # Map Raw IDs (100, 200) to Model IDs (0, 1)
        mask = self.encode_mask(mask)

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]

        return image, torch.tensor(mask, dtype=torch.long)

# --- TRAINING SETUP ---
def get_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1], interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def train_model():
    # Extraction Check
    if not os.path.exists(EXTRACT_PATH):
        print("Extracting Zip...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as z: z.extractall(EXTRACT_PATH)
    else:
        print("✅ Data already extracted.")

    dataset = OffroadMulticlassDataset(EXTRACT_PATH, transform=get_transforms())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    print(f"\n🚀 Multi-Class Training Started on {DEVICE}...")
    print(f"Classes: {NUM_CLASSES}")
    
    # Model: Unet++ with 10 Output Classes
    model = smp.UnetPlusPlus(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=NUM_CLASSES, 
        activation=None
    ).to(DEVICE)

    # Loss Function for Multi-Class
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda') 
    
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0
        
        for images, masks in loop:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                predictions = model(images)
                loss = loss_fn(predictions, masks)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_loss = epoch_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            loop.write(f"💾 Saved Best Model: {avg_loss:.4f}")

if __name__ == "__main__":
    train_model()