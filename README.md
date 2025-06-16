# Face Detection & Signature Recognition Toolkit

This project combines real-time face detection using a YOLO-based model and signature verification using custom feature extraction + ML classifiers. It includes model training, webcam inference, and 2D visualization of extracted features.

---

## Features

### Face Detection (YOLO + PyTorch)

* YOLO-style bounding box training with `ssdlite320_mobilenet_v3_large`
* Dataset support in **YOLO format** (txt label files)
* Live webcam inference with bounding boxes
* Mean Average Precision (mAP) evaluation

### Signature Verification

* Image preprocessing: grayscale, binarization, segmentation, thinning
* Custom feature extraction: Hu Moments, contour metrics, pixel density
* Classification with `k-NN` and `MLP`
* Support for binary (genuine/forgery) and multi-class (per user) labels
* Visualizations: PCA and t-SNE of signature embeddings

---

## Requirements

```bash
pip install torch torchvision opencv-python numpy matplotlib seaborn scikit-learn torchmetrics kagglehub joblib
```

> Also requires OpenCV's extra modules for `cv2.ximgproc.thinning`.

---

## Project Structure

```
project/
├── main.py                    # Entry point for face detection pipeline
├── config.py                  # Configurations for paths, thresholds, transforms
├── dataset.py                 # YOLO-format dataset loader
├── training.py                # Model training + evaluation logic
├── webcam.py                  # Real-time webcam inference
├── signature_recognition.py  # Signature verification pipeline
├── models.py                  # k-NN and MLP classifier logic
├── visualization.py          # PCA & t-SNE visualizations
├── utils.py                  # Project structure validator
```

---

## 1. Face Detection

### Dataset Format (YOLO style)

```
project/
├── images/
│   ├── train/
│   └── validation/
├── labels/
│   ├── train/
│   └── validation/
```

Each `.txt` file in `labels/` should follow this format:

```
class_id center_x center_y width height
```

All values should be normalized between 0 and 1.

### Training

Edit `config.py` to set:

```python
enable_training = True
```

Then run:

```bash
python main.py
```

After training, the best model is saved to:

```
face_checkpoints/best_yolo_face_model.pth
```

### Live Webcam Detection

After training (or loading a checkpoint), the script will ask:

```
Do you want to start live webcam face detection? (y/n)
```

* Press `p` to pause/resume
* Press `q` to quit

---

## 2. Signature Verification

Uses the **Kaggle** dataset: [`robinreni/signature-verification-dataset`](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset)

### Run

```bash
python signature_recognition.py
```

This will:

* Download the dataset
* Preprocess and extract features
* Train both `k-NN` and `MLP` classifiers
* Evaluate performance
* Generate:

  * `confusion_matrix_binary.png`
  * `pca_visualization.png`
  * `tsne_visualization.png`

### Features Extracted

* Black pixel count
* Aspect ratio
* Fill ratio
* Hu moments
* Contour counts
* Pixel distributions in image quadrants

---

## Visualizations

* `PCA`: Projects feature space into 2D based on variance
* `t-SNE`: Preserves local structure and clusters visually

Files saved:

```
pca_visualization.png
tsne_visualization.png
```

---

## Model Files

After running signature verification:

* `mlp_model_binary.pkl`, `scaler_binary.pkl`
* `mlp_model_multi.pkl`, `scaler_multi.pkl`

These can be reused for future predictions without retraining.

---

## Configuration

All paths, thresholds, and batch sizes can be configured in `config.py`.

---

## License

This project is intended for **educational** and **research** purposes only. Not suitable for production without further optimization and validation.

---
