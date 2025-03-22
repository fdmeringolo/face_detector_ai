# Face Detection from Scratch using AI

## Project Overview
This project implements a face detection system from scratch without using pre-trained models or algorithms. The system is capable of detecting human faces in images and providing a bounding box around the detected face. The entire implementation is contained within a Jupyter Notebook.

## Objectives
- Build a face recognition system without relying on pre-trained models.
- Train a custom face detector from scratch.
- Provide bounding box coordinates for detected faces.

## Dataset
The model is trained using the "Nonface and Face Dataset," which can be downloaded from Kaggle:
[Nonface and Face Dataset](https://www.kaggle.com/datasets/sagarkarar/nonface-and-face-dataset/data)

## Model and Training
- The model is trained using Support Vector Machines (SVC) and other machine learning techniques.
- It learns to distinguish between faces and non-faces using the dataset mentioned above.
- After training, the model is saved and available in: **`face_detector_model_SVC.pkl`**.

## Project Structure
- **Jupyter Notebook**: Contains the entire implementation, including data preprocessing, model training, and evaluation.
- **Dataset**: Images used for training and testing.
- **Trained Model**: The trained model is saved as `face_detector_model_SVC.pkl` for later use.
## Usage
To use this project, follow these steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/fdmeringolo/face_detector.git
    cd face_detector
    ```
2. Download the dataset from the Kaggle link above and place it in the `data/` directory.
3. Install the required dependencies:
    ```bash
    pip install numpy pandas scikit-learn matplotlib opencv-python jupyter joblib scikit-image
    ```
4. Open the Jupyter Notebook and execute the cells step by step to train the model.
5. Once trained, use the model to detect faces in new images.

## Requirements
To run this project, install the following dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib opencv-python jupyter joblib scikit-image
```

## Future Improvements
- Extend the model to detect multiple faces in an image.
- Improve accuracy using deep learning techniques.
- Optimize performance for real-time face detection.

## License
This project is open-source and available for contributions under the MIT License. See the LICENSE file for more details.

---
Feel free to modify and expand this README as needed!

