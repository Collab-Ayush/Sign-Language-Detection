# Sign Language Detection using CNN and PyTorch

## Overview

This project is a real-time **Sign Language Detection System** built using **PyTorch**, **OpenCV**, and a **Convolutional Neural Network (CNN)** trained on the **Sign Language MNIST Dataset**.

The model predicts hand signs from webcam input and displays the corresponding alphabet letter in real time.

The project consists of:

* Training a CNN model using the Sign Language MNIST dataset
* Saving the trained model weights
* Performing real-time sign detection using a webcam
* Predicting alphabet gestures live on screen

---

## Features

* Real-time webcam-based sign detection
* CNN-based deep learning architecture
* Trained using Sign Language MNIST dataset
* Live alphabet prediction display
* ROI (Region of Interest) based hand detection
* GPU support with CUDA (if available)

---

## Tech Stack

* Python
* PyTorch
* OpenCV
* NumPy
* Pandas
* Google Colab

---

## Project Structure

```bash
.
├── SIgnLanguage_final.ipynb     # Training notebook
├── Live_detection.py            # Real-time prediction script
├── sign_mnist_cnn.pth           # Trained model weights
└── README.md
```

---

## Dataset

This project uses the **Sign Language MNIST Dataset** from Kaggle.

Dataset Link:

[https://www.kaggle.com/datasets/datamunge/sign-language-mnist](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

The dataset contains grayscale images of hand gestures representing alphabet letters.

> Note:
> The dataset excludes the letters **J** and **Z** because they require motion.

---

## CNN Architecture

The CNN model consists of:

* 2 Convolutional Layers
* ReLU Activation
* Max Pooling Layers
* Dropout Layer
* Fully Connected Dense Layers

### Architecture Summary

```python
Conv2D(1 → 32)
ReLU
MaxPool

Conv2D(32 → 64)
ReLU
MaxPool

Flatten
Dropout(0.3)
Linear(3136 → 128)
ReLU
Linear(128 → 26)
```

---

## Model Training

The model was trained using:

* Loss Function: CrossEntropyLoss
* Optimizer: Adam
* Epochs: 15
* Learning Rate: 0.001

Training was performed in Google Colab.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sign-language-detection.git
cd sign-language-detection
```

### 2. Install Dependencies

```bash
pip install torch torchvision opencv-python numpy pandas
```

---

## Running the Project

### Step 1: Train the Model (Optional)

Open the notebook:

```bash
SIgnLanguage_final.ipynb
```

Train the model and generate:

```bash
sign_mnist_cnn.pth
```

---

### Step 2: Run Real-Time Detection

```bash
python Live_detection.py
```

---

## How It Works

1. Webcam captures live video.
2. A Region of Interest (ROI) is drawn on screen.
3. The hand gesture inside the ROI is processed.
4. Image preprocessing includes:

   * Grayscale conversion
   * Gaussian blur
   * Resizing to 28×28
   * Normalization
5. The processed image is passed to the CNN model.
6. The predicted alphabet letter is displayed live.

---

## Real-Time Detection Preview

When the program runs:

* A webcam window opens.
* Place your hand inside the green box.
* The model predicts the corresponding alphabet.
* Press `q` to quit.

---

## Future Improvements

* Add support for dynamic gestures (J and Z)
* Improve accuracy using advanced architectures
* Add sentence formation capability
* Deploy as a web application
* Add voice output for predictions
* Use MediaPipe for better hand tracking

---

## Example Output

```bash
Prediction: A
Prediction: B
Prediction: C
```

---

## Requirements

```txt
torch
opencv-python
numpy
pandas
```

---

## Author

Ayush Dutta

---

## License

This project is under MIT License

---

## Acknowledgements

* Kaggle Sign Language MNIST Dataset
* PyTorch Documentation
* OpenCV Documentation

---

## GitHub Upload Steps

### Initialize Git

```bash
git init
```

### Add Files

```bash
git add .
```

### Commit

```bash
git commit -m "Initial commit"
```

### Connect Repository

```bash
git remote add origin https://github.com/Collab-Ayush/Sign-Language-Detection.git
```

### Push to GitHub

```bash
git branch -M main
git push -u origin main
```
