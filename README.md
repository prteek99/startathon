# 🌵 Offroad Driving AI - Desert Scene Segmentation

This project is a smart AI system designed for off-road vehicles. It helps robot vehicles "see" and understand desert environments by identifying trees, rocks, sand, and sky in real-time. This was built for the **Duality AI Hackathon**.

## 📊 1. Final Results (How well did the AI do?)

Our AI is very good at identifying objects. We measured it using a score called **mAP50** (which tells us how precise the AI is).

| Metric | Score | Percentage |
| :--- | :--- | :--- |
| **Precision (mAP50)** | **0.8037** | **80.37%** |
| Accuracy (mIoU) | 0.6151 | 61.51% |
| Recall (Finding objects) | 0.7025 | 70.25% |

> [!NOTE]
> **Final Mean IoU recorded during evaluation: 0.6156**
<img width="1918" height="977" alt="image" src="https://github.com/user-attachments/assets/6dc5f219-3056-471f-90e7-516e19121963" />

---

## 💾 2. Download the Trained AI
Because the AI's "brain" file is quite large (90MB), it is stored on Google Drive. You need this file to run the project.

* **File Name:** `best_multiclass_model.pth`
* **Size:** 90 MB
* **Brain Type:** U-Net++ (ResNet34)

[**➡️ Click here to Download the Model Weights**](https://drive.google.com/file/d/1cv1gDyeddIbIQz6ymJMULvLkTbyRNFWN/view?usp=sharing)

---

## 🚀 3. Our Journey (The Log Book)

Building this AI wasn't easy! Here is what we did step-by-step to make it better.

### 📁 Getting the Data Ready
* **The ID Problem:** The dataset used weird numbers to identify objects (like 100 for a tree). The AI couldn't understand this.
  * **Solution:** We built a "Translator" (Class Mapper) that changed those numbers into simple labels from 0 to 9.
* **The "Black Screen" Mystery:** At first, all our data looked like solid black boxes. We realized the data was "16-bit," which is too high-quality for basic photo viewers.
  * **Solution:** We changed the code to read the high-quality files properly so the AI could finally "see" the training images.

### 🧠 Training the AI
* **Learning Small Objects:** Early on, the AI was great at seeing the big sky but kept missing small **Rocks** and **Logs**.
  * **Solution:** We changed the "Loss Function" (the AI's teacher). We told the teacher to give the AI extra "homework" specifically on small, hard-to-see objects.
* **Handling Light & Shadows:** The AI was getting confused by different lighting in the desert.
  * **Solution:** We used "Data Augmentation." We intentionally blurred some photos and changed the brightness so the AI learned to work in any weather.

---

## 🔍 4. Fixing Mistakes (Failure Analysis)

### Case 1: Hidden Rocks
* **Problem:** If a rock was half-covered by dry grass, the AI ignored it.
* **The Fix:** We adjusted the training settings to be more "sensitive" to textures, helping the AI distinguish between grass and stones.

### Case 2: The "Solid Red" Error
* **Problem:** During our first tests, the output images looked like solid red blocks.
* **The Fix:** We realized the computer was reading the data in "Color" when it should have been "Black & White." Once we fixed the setting, the colors came back perfectly.

---

## 🛠️ 5. How to Run This Project

### 📦 What you need
Make sure you have Python installed, then run this command to get the tools:
`pip install torch segmentation-models-pytorch albumentations opencv-python matplotlib tqdm`

### 🎮 How to use
1. **To Test:** Put your images in the folder and run `python test.py`. It will show you a comparison of the Original vs. the AI's Prediction.
2. **To Train:** If you want to train the AI again from scratch, run `python train.py`.

### 🏗️ Technical Specs
* **AI Architecture:** U-Net++ (ResNet34 Encoder)
* **Training Time:** 44 Rounds (Epochs)
* **Speed:** Very Fast (< 50ms per image)
