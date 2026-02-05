# SMART-ROAD-SIGN-INTERACTION-SYSTEM-USING-COMPUTER-VISION-AND-CAMERAS
A real-time traffic sign recognition system built with Convolutional Neural Networks (CNN) achieving 99.63% validation accuracy. The system includes a structured preprocessing pipeline and an efficient CNN architecture to generalize well on unseen real-world images.

## 🧠 Project Overview

The system uses a **Convolutional Neural Network (CNN)** trained on a well-curated dataset to accurately identify traffic signs under diverse conditions. It provides a **user-friendly Gradio interface** for real-time testing and visualization of predictions.  

Key achievements of this project include:

- **Validation Accuracy:** 99.63%  
- **Loss:** 0.0032  
- Robust performance on **unseen real-world images**  
- Efficient, low-resource CNN architecture suitable for real-time deployment  

---

## 🛠 Features

- Real-time traffic sign detection  
- High accuracy with minimal computational requirements  
- User-friendly **Gradio interface** for testing predictions  
- Generalizes well to various lighting, orientation, and environmental conditions  
- Supports intelligent transportation systems and autonomous vehicles  

---

## 📦 Dataset

The dataset used is the **German Traffic Sign Recognition Benchmark (GTSRB)**:

- Over **50,000 images** of traffic signs  
- **6 classes** used in this project:  

```python
classes = {
 0:'Speed limit (20km/h)',
 1:'Speed limit (30km/h)',
 2:'Stop',
 3:'No entry',
 4:'Turn right ahead',
 5:'Turn left ahead'
}

```
## 🧹 Image Preprocessing and Segmentation

To prepare images for the CNN model, the following preprocessing steps were applied:

1. **Resize:** Images standardized to **32x32 pixels** using `cv2.resize()`.  
2. **Grayscale Conversion:** Images converted to grayscale with `cv2.cvtColor()` to reduce computational complexity.  
3. **Histogram Equalization:** Applied using `cv2.equalizeHist()` to enhance contrast and highlight features.  
4. **Normalization:** Pixel values scaled to [0, 1] for neural network input.  

This preprocessing ensures uniformity and improves the model’s ability to learn relevant features.
---

## 🧠 Model Architecture

- **Convolutional Neural Network (CNN)**  
- **Input:** 32x32 grayscale images  
- **Layers:** Convolutional, MaxPooling, Flatten, Dense layers  
- **Output:** 6-class softmax classification  
- Trained using the preprocessed GTSRB dataset for high accuracy and minimal loss  

---

## 📊 Evaluation

- **Validation Accuracy:** 99.63%  
- **Loss:** 0.0032  
- The model generalizes well on unseen images, indicating strong predictive reliability  

---

## 🖼 Testing & Visualization

The system provides a **Gradio interface** for real-time predictions. Users can upload an image and immediately view the predicted traffic sign.  

Example visualization in Python:

```python
plt.imshow(img[0, :, :, 0], cmap="gray")
plt.title(f"Detected Traffic Sign: {classes[predicted_class]}", fontsize=12, color="red")
plt.axis("off")
plt.show()
```
## 🐍 How to Run

- Clone the repository:

```
git clone https://github.com/Yagnathanakanti/Traffic-Sign-Recognition.git
cd Traffic-Sign-Recognition

```
- Install dependencies:

```pip install -r requirements.txt```


- Run the notebook or Python script:

```python traffic_sign_detection.py```


- For real-time testing, launch the Gradio interface:

```python gradio_interface.py```

## 🛠 Technologies Used

- Python, OpenCV, NumPy, Matplotlib

- TensorFlow, Keras

- CNN for image classification

- Gradio for interactive testing

## References

- Link - https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
