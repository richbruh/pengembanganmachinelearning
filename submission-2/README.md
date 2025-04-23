# Apple Image Classification Project

## Overview
This project implements a Convolutional Neural Network (CNN) to classify images of fruits, specifically focused on different varieties of apples, peaches, and pears. The model is trained on the Fruits-360 dataset and achieves high accuracy in distinguishing between different fruit varieties.

## Author
- **Name:** Richie Rich Kennedy Zakaria
- **Email:** richie.zakaria100@gmail.com
- **ID:** mc271d5y0626

## Dataset
Project ini menggunakan [Fruits-360 dataset](https://www.kaggle.com/datasets/moltean/fruits) dari kaggle yang berfokus pada:
- Multiple varieties of apples
- Multiple varieties of pears
- Multiple varieties of peaches

## SPECIAL NOTE: IKUTI INSTRUKSI YANG ADA PADA NOTEBOOK.IPYNB 

Split 80% dan 20 %

## Project Structure
```
submission/
├── saved_model/               # TensorFlow SavedModel format
│   ├── saved_model.pb
│   └── variables/
├── tflite/                    # TensorFlow Lite format
│   ├── model.tflite
│   └── label.txt              # Class labels for the model
├── tfjs_model/                # TensorFlow.js format
│   ├── model.json
│   └── group1-shard1of1.bin
├── notebook.ipynb             # Main Jupyter notebook with all code
├── README.md                  # This file
└── requirements.txt           # Required Python packages
```

## Requirements
The project requires the following Python packages:
- tensorflow >= 2.4.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- tensorflowjs >= 3.0.0
- kaggle >= 1.5.12
- pillow >= 8.0.0

You can install all requirements using:
```
pip install -r requirements.txt
```

## Model Architecture
The model uses a CNN architecture with the following layers:
- 3 Convolutional layers with ReLU activation and MaxPooling
- Flatten layer
- Dense layer with 512 units and ReLU activation
- Dropout layer (0.5) for regularization
- Output layer with softmax activation (number of units equals the number of fruit classes)

## Training Process
The model was trained with the following specifications:
- Data augmentation: rotation, width/height shifts, shear, zoom, and horizontal flip
- Optimizer: Adam
- Loss function: Categorical Cross-Entropy
- Early stopping: Based on validation loss with patience of 5 epochs
- Model checkpoint: Saving the best model based on validation accuracy

## Model Formats
The trained model is provided in three different formats:

1. **SavedModel**
   - Standard TensorFlow format
   - Located in the `saved_model` directory

2. **TensorFlow Lite**
   - Optimized for mobile and edge devices
   - Located in the `tflite` directory along with class labels

3. **TensorFlow.js**
   - For deployment in web browsers
   - Located in the `tfjs_model` directory

## Cara penggunaan
1. Clone repository
2. Install requirements: `pip install -r requirements.txt`
3. Open dan run all `notebook.ipynb` to see the full project workflow

### Inference Example
```python
# Using SavedModel format
import tensorflow as tf
model = tf.keras.models.load_model('saved_model')

# Using TFLite format
interpreter = tf.lite.Interpreter(model_path="tflite/model.tflite")
interpreter.allocate_tensors()
```

