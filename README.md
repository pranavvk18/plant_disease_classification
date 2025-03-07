# Plant Disease Detection Project

## Introduction
This project aims to develop a machine learning model for plant disease detection using image classification techniques. By leveraging deep learning, the model can identify various plant diseases from leaf images, helping farmers and agricultural experts take timely action.

## Dataset
We use the **New Plant Diseases Dataset** sourced from Kaggle, which consists of labeled images of healthy and diseased plants.
* **Source:** Kaggle - New Plant Diseases Dataset
* **Categories:** Healthy and Diseased plants
* **Format:** JPEG images

## Dataset Structure
The dataset is structured into training and validation sets:

```
├── train
│   ├── Class1
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   ├── Class2
│
├── validation
│   ├── Class1
│   ├── Class2
```

## Project Setup
### Requirements
To run this project, install the following dependencies:

```
pip install tensorflow keras opencv-python numpy matplotlib
```

## Training the Model
1. **Download the dataset** from Kaggle and extract it.
2. **Preprocess the images** (resizing, normalization, augmentation if needed).
3. **Load the dataset** using TensorFlow or PyTorch.
4. **Train a classification model** using CNNs or Transfer Learning.

### Example Code (Loading Dataset in Python)

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path = "path_to_extracted_dataset"

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    dataset_path + "/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    dataset_path + "/validation",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

## Applications
* Automated plant disease detection
* Agricultural health monitoring
* Precision farming using AI

## Contributing
Feel free to contribute to this project by improving the model, optimizing performance, or expanding the dataset.
