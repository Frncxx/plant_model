# Plant Disease Detection
A deep learning model for classifying plant diseases using leaves images. The model classifies the images into three categories: Healthy, Powdery and Rust.

## Overview
Using a pre-trained model, MobileNetV3Large, with Transfer Learning in tensorflow and tensorflow-lite to classify leaves images into the three categories.Then, applied fine-tuning with the model on the dataset.

## Dataset Collection
- Our final Dataset consisted and collected from 3 different datasets.
- The latest version of dataset contains three categories: Healthy, Powdery and Rust.
- The first version, each class was about 400 images.
- The second version we used [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) , for the full research paper refer [Arxiv](https://arxiv.org/abs/1911.10317) and cleaned the data from noisy (unwanted) images. This allowed us to increase the images to be about 650 images per each class.
- The third, latest, version we used a dataset from Hugging Face website called [Plant Disease Recognition](https://huggingface.co/datasets/NouRed/plant-disease-recognition) which helped to increase the dataset to about 1000 images in each class.

## Requirements
- NumPy
- Pillow
- Streamlit
- Tensorflow
- Matplotlib

## Training & Data Augmentation
- Loaded the dataset into train, validation and test datasets.
- Applied different data augmentation techniques as Sequential Keras Model:
  - RandomFlip.
  - RandomRotation.
  - RandomBrightness.
  - RandomContrast.
  - RandomZoom.
- Preprocessing images using `preprocess_input` to normalize values.

## Evaluation and Callbacks
- The evaluation metrics are accuracy and loss.
- We used Callbacks like `Early Stopping` to stop training when monitored metric, validation loss has stopped improving to avoid model overfitting and `ReduceLROnPlateau` to reduce the base learning rate we used.

## Streamlit Web App
A streamlit web application to deploy the model and make project more easily to be used by user.
you can access the web app from below.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_red.svg)](https://plant-leaf-disease-detection.streamlit.app/)

## Project Demo Video
**SOON**

## Usage
#### 1- Clone the repository:
```
git clone https://github.com/AhmAshraf1/plant_model.git
cd plant_model
```

#### 2- Install dependencies:
```
pip install -r requirements.txt
```

### 3- Run Streamlit App:
```
streamlit run /app/app.py
```

## Results
The model achieves an accuracy of 96% on the test set. Training and Validation loss and accuracy plots are provided in the Notebook to visualize the model's performance after transfer learning and fine-tuning. Confusion Matrix and Classification Report are also provided in the notebook to show the classification performance on the test set.