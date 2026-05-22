# Wafer Map Defect Finder

An AI tool built with PyTorch to automatically detect and classify manufacturing flaws on microchip wafer maps. Reengineered from a classic ResNet image recognition model, it processes raw factory data instead of camera photos. It uses smart data tricks to fix heavy class imbalances, hitting a clean **95% accuracy** on unseen test batches.

---

## 🚀 Performance Highlights
* **Final Test Accuracy:** `95.00%`
* **Final Test Loss:** `0.1402`
* **Generalization Strength:** Reached high operational stability by using the updated `torchvision.transforms.v2` engine to execute on-the-fly, high-speed geometric shifts directly in system memory.

---

## 🧬 Model Architecture

The core of this project is a custom-built **ResNet-18** neural network designed specifically for industrial wafer maps. While standard image-recognition models are built for colorful 3-channel (RGB) smartphone photos, semiconductor data consists of single-channel factory grids. 

To bridge this gap, the network was rebuilt from scratch to process raw manufacturing data efficiently without dropping accuracy.

### Structural Flow Diagram

[Raw Wafer Data: 1 x 100 x 100]
│
▼
[Custom Input Layer (Stem)]   ──► Reconfigured to read 1-channel data instead of 3-channel RGB.
│
▼
[Residual Layer 1 Stack]    ──► 2x Basic blocks processing initial low-level shape patterns.
│
▼
[Residual Layer 2 Stack]    ──► 2x Blocks with a 1x1 shortcut layer to downsample spatial data.
│
▼
[Residual Layer 3 Stack]    ──► 2x Blocks tracking mid-level geometric distributions.
│
▼
[Residual Layer 4 Stack]    ──► 2x Blocks extracting high-level semantic defect structures.
│
▼
[Global Average Pooling]     ──► Flattens the deep 512-feature maps into a compact numerical vector.
│
▼
[Linear Output Classifier]   ──► Maps the finalized features to predict 1 of the 9 defect classes.

### Key Network Optimizations

* **Grayscale Input Adaptation:** The initial convolutional layer was re-engineered with `in_channels=1` to native-match the dimensions of your wafer arrays, preventing memory waste.
* **Smart Identity Shortcuts:** Reconfigured the internal 1x1 shortcut connections to scale down dynamically with the layer strides. This keeps the gradient highway intact and completely prevents structural "shape mismatch" crashes.
* **Lean Resource Footprint:** By removing standard convolutional biases (`bias=False`) right before Batch Normalization layers, the network avoids dead computing cycles and trains significantly faster on the GPU.

---

## 🛠️ Optimization Pipeline

The pipeline uses advanced optimization methods designed for high-class imbalance and smooth local minima convergence:

* **Optimizer:** `AdamW` (Weight Decay: `1e-4`, Baseline Learning Rate: `1e-3`)
* **Learning Rate Scheduler:** `ReduceLROnPlateau` (Dynamic scale factor: `0.5`, patience: `2 epochs` tracking validation loss)
* **Imbalance Correction:** Handled intense factory distribution skews by injecting customized inverse-frequency cross-entropy loss weight penalties across all 9 target classes.
## To read the Jupyter notebook, open the .ipynb file and in the url change the github to "githubtocolab"
