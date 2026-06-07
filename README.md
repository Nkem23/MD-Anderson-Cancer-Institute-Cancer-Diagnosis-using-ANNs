## Cancer Diagnosis Using Artificial Neural Networks at MD Anderson Cancer Institute

## Project Overview
This project implements an Artificial Neural Network (ANN) using TensorFlow and Keras 
to classify breast tumours as Benign or Malignant. It was developed as part of the 
BAN6440 Applied Machine Learning for Business Analytics course at Nexford University, 
simulating a real-world diagnostic support tool for MD Anderson Cancer Institute.

## Dataset
- **Source:** Wisconsin Breast Cancer Dataset (via Kaggle)
- **Records:** 569 patients
- **Features:** 30 numerical cell measurement features
- **Target:** Binary diagnosis: Benign (B) or Malignant (M)
- **Missing Values:** None
- **Class Distribution:** 62.7% Benign, 37.3% Malignant

## Project Structure
├── ANN_Cancer_Diagnosis_MDA.ipynb  : Main Jupyter Notebook

├── Cancer_Ann_Diagnosis.py         : Python Script

├── README.md

├── breast_cancer_data.csv          : Dataset

├── cancer_ann_model.keras          : Saved trained model

├── confusion_matrix.png            : Accuracy and loss over training cycles

├── training_curves.png             : Model prediction breakdown



## Model Architecture
| Layer | Type | Neurons | Activation |
|-------|------|---------|------------|
| 1 | Dense (Hidden) | 16 | ReLU |
| 2 | Dropout | 30% | - |
| 3 | Dense (Hidden) | 8 | ReLU |
| 4 | Dense (Output) | 1 | Sigmoid |

- **Total Parameters:** 641
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Improvement Applied:** Dropout (0.3) + Early Stopping (patience=10)

## Results
| Metric | Benign | Malignant | Overall |
|--------|--------|-----------|---------|
| Precision | 0.97 | 1.00 | 0.98 |
| Recall | 1.00 | 0.95 | 0.98 |
| F1-Score | 0.99 | 0.98 | 0.98 |
| Accuracy | - | -| 98.25% |

## Key Insight
In cancer diagnosis, **recall for Malignant cases is the most critical metric** 
missing a real cancer case carries far greater risk than a false alarm. 
Our model achieved 95% recall on Malignant cases, catching 40 out of 42 actual 
cancer patients in the test set.

## Tools and Libraries
- Python 3
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

## Author
Adaora Nkemdilim Okafor | Nexford University | BAN6440 Module 5


