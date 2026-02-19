<img width="1918" height="977" alt="image" src="https://github.com/user-attachments/assets/459246ad-c9da-4a99-9ffc-fd78c79c8ce5" />

1. Methodology & Workflow Log
Here is a summary of the major challenges we faced and how we solved them to improve our model's performance.

Entry 1: Data Preparation

Task: Setting up the dataset for the model.

Issue Faced: The dataset used non-standard Class IDs (e.g., 100 for Trees, 10000 for Sky). The standard AI model expected sequential numbers (0, 1, 2...), causing errors during the start of training.

Solution: We wrote a custom Class Mapper function. This function converts the raw IDs (100, 200, etc.) into a 0-9 format before feeding them into the model.

Entry 2: Handling High-Bit Depth Masks

Task: Loading Ground Truth Masks.

Issue Faced: The training masks appeared completely black, and the model was not learning anything. This happened because the masks were 16-bit images, but standard image loaders were reading them as 8-bit, losing all the data.

Solution: We modified the image loading code to use cv2.imread(path, -1). This flag forced the code to read the raw, unchanged 16-bit data, revealing the hidden class IDs.

Entry 3: Initial Training Performance

Task: Baseline Model Training (Epochs 1-10).

Initial IoU Score: ~0.58

Issue Faced: The model achieved high accuracy on large classes (Sky, Landscape) but performed poorly on small, difficult objects like "Logs" and "Rocks" (Low Recall).

Solution: We changed the Loss Function. Instead of simple CrossEntropy, we implemented a combination of Dice Loss + Focal Loss. Focal Loss forces the model to focus harder on small, difficult-to-classify objects.

Entry 4: Improving Generalization

Task: Final Model Optimization (Epochs 10-40).

Issue Faced: The model started overfitting (memorizing) the training data and struggled with lighting changes in the validation set.

Solution: We introduced Heavy Data Augmentation. We added random rotations, color jitter (brightness/contrast changes), and Gaussian blur to the training pipeline. This helped the model learn to recognize objects even in different environmental conditions.

Entry 5: Visualization Bug Fix

Task: Generating Final Submission Images.

Issue Faced: When generating the comparison report, the "Ground Truth" images appeared as solid red/orange blocks. The system was misinterpreting the mask channels (RGB vs. Grayscale).

Solution: We updated the testing script to strictly enforce 2D (1-Channel) reading of masks. This corrected the visualization, allowing us to accurately compare predictions against the ground truth.

2. Failure Case Analysis
(Note for you: In your report, paste the images we generated in Test_Visualizations folder that match these descriptions)

Case 1: Small Object Occlusion

Observation: The model initially failed to detect "Rocks" when they were partially hidden by "Dry Grass."

Why it happened: The texture of dry grass and small rocks is very similar in color.

Fix: We increased the weight of the Focal Loss function, which penalizes the model more heavily for missing these hard examples.

Case 2: The "Solid Red" Visualization Error

Observation: During testing, the Ground Truth masks appeared as a single solid color (Orange/Red), making evaluation impossible.

Why it happened: The code was reading the mask file as a 3-channel color image, but the class ID data was only present in a single channel.

Fix: We modified the code to explicitly read the masks in Grayscale Mode, ensuring the class IDs were preserved correctly.

3. Final Results Summary
Model Architecture: U-Net++ (ResNet34 Encoder)

Total Epochs: 40

Loss Function: DiceLoss + FocalLoss

Final IoU Score: [Insert your final score from the terminal here, e.g., 0.72]

Inference Speed: < 50ms per image (Verified on GPU)
