# COVID-19 Chest X-Ray Classification & Explainable AI (XAI)

This repository contains a deep learning framework for classifying chest X-ray images into three distinct categories (COVID-19, Normal, and Viral Pneumonia) using **DenseNet-121** and **Vision Transformers (ViT)**, augmented with **Explainable AI (XAI)** techniques to visually explain model predictions.

Deep learning models in medical imaging are often perceived as "black boxes." This project implements explainability methods like **Grad-CAM**, **Integrated Gradients**, and **ViT Attention Rollout** to highlight the key pathological regions within the chest X-rays that drove the models' decisions, helping clinicians build trust in the machine learning outputs.

---

## 📁 Project Structure

The project has been organized with the following directory structure:

```text
COVID19-XRay-Classification-XAI/
│
├── README.md                 # Project documentation and summary
├── Aims.ipynb                # Main Jupyter Notebook with data loaders, training, and XAI
├── requirements.txt          # Python dependencies list
│
├── models/
│   └── best_densenet121.pth  # Saved weights for the best-performing DenseNet121 model
│
└── outputs/
    ├── confusion_matrix.png       # Test set confusion matrix
    ├── gradcam_result.png         # Grad-CAM visualization overlay
    └── integrated_gradients.png   # Integrated Gradients visualization overlay
```

---

## 🧠 Model Architectures & Performance

Two state-of-the-art architectures were trained and evaluated on a chest X-ray dataset of 4,035 images:
1. **DenseNet-121**: A convolutional neural network designed for dense feature propagation and reuse, pre-trained on ImageNet.
2. **Vision Transformer (ViT-Base)**: A transformer-based model using self-attention mechanisms to learn global context relationships in image patches.

### DenseNet-121 Performance
The DenseNet-121 classifier achieved exceptional results on the test set:
* **Accuracy**: **99.54%**
* **Precision**: **99.55%**
* **Recall**: **99.54%**
* **F1-Score**: **99.54%**

#### Classification Report:
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **COVID** | 1.00 | 0.99 | 0.99 | 145 |
| **Normal** | 0.99 | 1.00 | 0.99 | 145 |
| **Viral Pneumonia** | 1.00 | 1.00 | 1.00 | 145 |

#### Test Set Confusion Matrix:
![Confusion Matrix](outputs/confusion_matrix.png)

---

## 🔍 Explainable AI (XAI) Methods & Interpretability

To interpret what features the models use to make decisions, several explainability methods were implemented and compared:

### 1. Grad-CAM (Gradient-weighted Class Activation Mapping)
Grad-CAM uses the gradients of any target concept flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image.
* **Target Layer**: Last dense block of DenseNet121 (`model.features[-1]`).
* **Visual Output**: Heatmap highlighting pathological areas, such as ground-glass opacities, overlaid on the chest X-ray.

![Grad-CAM Result](outputs/gradcam_result.png)

### 2. Integrated Gradients
Integrated Gradients is a path-attribution method that computes the path integral of the gradients of the output with respect to the input along a straight path from a baseline (blank canvas) to the input image. It satisfies key axioms like *Completeness* and *Implementation Invariance*.
* **Visual Output**: Fine-grained pixel-level importance attributes shown as a jet-colored overlay.

![Integrated Gradients Result](outputs/integrated_gradients.png)

### 3. Vision Transformer (ViT) Attention Rollout
ViT Attention Rollout computes how information flows through the self-attention layers from the input patches to the final representation.
* **Visual Output**: Fused self-attention rollout heatmap highlighting the patches the model attended to.

### 📊 Quantifying Explainability
The XAI heatmaps were evaluated using standard interpretability metrics:
* **Entropy**: Measures the concentration of the attribution map (lower means more localized focus).
* **Insertion**: Evaluates how prediction probability changes when pixels are added in order of importance.
* **Deletion**: Evaluates how prediction probability drops when pixels are deleted in order of importance (lower deletion score/higher deletion rate represents better fidelity).
* **AOPC (Area Over the Perturbation Curve)**: Quantifies the impact of perturbing critical pixels.

| Method | Entropy | Insertion | Deletion | AOPC |
| :--- | :---: | :---: | :---: | :---: |
| **Grad-CAM** | 10.67 | 0.542 | 0.458 | 0.542 |
| **Integrated Gradients** | 10.37 | 0.042 | 0.958 | 0.042 |

---

## 🛠️ Setup & Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/COVID19-XRay-Classification-XAI.git
   cd COVID19-XRay-Classification-XAI
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Install dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify the Model File**:
   Make sure the pre-trained weights file `models/best_densenet121.pth` is present in the repository before running inference or XAI visualizations.

4. **Run the Notebook**:
   Launch Jupyter Notebook or JupyterLab to interactively explore the dataset loading, training, and explainability implementations:
   ```bash
   jupyter notebook Aims.ipynb
   ```
