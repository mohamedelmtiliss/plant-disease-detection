
# 🌿 Plant Disease Detection & Explainable AI

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

## 📌 About The Project
This project is an end-to-end Machine Learning solution for detecting plant leaf diseases (Tomato, Potato, and Bell Pepper) using **Deep Learning** and **Computer Vision**. 
Built on top of a fine-tuned **ResNet50** model, it features an interactive web application that provides instant disease classification, underlying causes, and agricultural treatment advice. Crucially, the project leverages **Explainable AI (XAI)** techniques to generate precise heatmaps, visually highlighting the exact infected spots on the leaves rather than relying on spurious background correlations.

## ✨ Key Features
* **High Accuracy (87%):** Robust classification across 15 different disease and healthy plant classes.
* **Explainable AI (XAI):** Implements **GradCAM++** combined with custom OpenCV masking to pinpoint and visualize the exact diseased lesions on the leaf at a pixel level.
* **Interactive Web App:** A user-friendly interface built with **Streamlit**, allowing users to upload leaf images and get real-time diagnostics.
* **Agricultural Knowledge Base:** Displays contextual agricultural advice, disease causes, and treatment methods for each specific prediction.
* **Robust Training Pipeline:** Mitigates catastrophic forgetting and overfitting using `Early Stopping`, layer freezing, and heavy data augmentation (`Color Jittering`, `Random Erasing`, `Random Rotation`).

## 📊 Dataset
The model was trained on a curated subset of the **PlantVillage** dataset, comprising 15 distinct classes:
* **Pepper (Bell):** Bacterial spot, Healthy.
* **Potato:** Early blight, Late blight, Healthy.
* **Tomato:** Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, YellowLeaf Curl Virus, Mosaic virus, Healthy.

## 🛠️ Tech Stack
* **Deep Learning Model:** PyTorch (ResNet50)
* **Image Processing:** OpenCV, Pillow, Torchvision Transforms
* **Explainable AI:** `pytorch-grad-cam` (GradCAMPlusPlus)
* **Web Deployment:** Streamlit
* **Evaluation & Metrics:** Scikit-learn (Confusion Matrix, Classification Report), Seaborn, Pandas, Matplotlib

## 🚀 Installation & Usage

1. **Clone the repository:**
```bash
git clone https://github.com/mohamedelmtiliss/plant-disease-detection
cd plant-disease-detection
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```


3. **Run the Streamlit Web App:**
```bash
streamlit run app.py
```


The application will automatically open in your default web browser at `http://localhost:8501`.

## 📈 Results & Evaluation

The fine-tuned ResNet50 model achieved an overall accuracy of **87%** on the validation set after 20 epochs.
Detailed evaluation using a Confusion Matrix and Classification Report showed exceptional precision (>95%) in classes like *YellowLeaf Curl Virus* and *Bacterial Spot*. The integration of **GradCAM++** successfully resolved initial padding artifacts and spurious background correlations, ensuring the model's predictions are genuinely based on the biological symptoms of the leaves.

## 🔮 Future Work

* **Object Detection:** Transitioning from Image Classification to Object Detection using **YOLOv8** to draw precise bounding boxes around diseased lesions.
* **AI Background Removal:** Integrating `rembg` into the data pipeline to automatically remove laboratory backgrounds from the PlantVillage dataset, forcing the model to focus purely on the plant's texture and symptoms.
