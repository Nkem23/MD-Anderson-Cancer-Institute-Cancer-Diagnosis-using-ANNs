## Cancer Diagnosis Using Artificial Neural Networks at MD Anderson Cancer Institute

### Overview
This project develops an Artificial Neural Network (ANN) using TensorFlow to support cancer diagnosis at MD Anderson Cancer Institute. The objective is to classify breast tumors as either benign or malignant based on clinical and radiological characteristics extracted from patient data.

Artificial Neural Networks are powerful machine learning models capable of learning complex patterns from data. In healthcare, these models can assist clinicians by providing predictive insights that support diagnostic decision-making and improve the efficiency of patient assessment processes.

This project demonstrates the complete machine learning workflow, including data preparation, model development, training, evaluation, and identification of potential improvements for future clinical applications.
Dataset

The project uses the Wisconsin Breast Cancer Diagnostic Dataset, obtained from a publicly available Kaggle repository and treated as a representative dataset provided by MD Anderson Cancer Institute for the purpose of this assignment.

### Dataset Characteristics
-  Total Patient Records: 569
-  Benign Cases: 357
-  	Malignant Cases: 212
-   Predictor Variables: 30 numerical features
-   Target Variable: Diagnosis (Benign = 0, Malignant = 1)
-   Missing Values: None

The features describe important cellular characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

###  Data Preprocessing
The following preprocessing steps were performed before model development:
-  Removal of non-clinical identifier columns
-  Data quality verification and missing value assessment
-	 Label encoding of the diagnosis variable
-	 Standardization of numerical features using StandardScaler
-	 Stratified train-test split (80% training, 20% testing)
-  Data leakage prevention through fitting transformations only on the training dataset
-  
These steps ensured that the data was suitable for neural network training and evaluation.

###  Artificial Neural Network Architecture
The model was developed using TensorFlow's Sequential API and consists of:
Layer	Configuration
Input Layer	30 Input Features
Hidden Layer 1	16 Neurons, ReLU Activation
Dropout Layer	20% Dropout Rate
Hidden Layer 2	8 Neurons, ReLU Activation
Output Layer	1 Neuron, Sigmoid Activation
The architecture was designed to balance predictive accuracy, computational efficiency, and model generalization.

###  Model Training Configuration
-  Optimizer: Adam
-	 Loss Function: Binary Cross-Entropy
-	 Batch Size: 32
-	 Epochs: 50
-	 Validation Monitoring: Accuracy and Loss
  
The model was trained to learn patterns associated with benign and malignant tumors while minimizing overfitting.

### Model Performance
The trained neural network achieved strong classification performance on previously unseen test data.
### Performance Metrics
•	Overall Accuracy: 97.37%
•	Malignant Precision: 100.00%
•	Malignant Recall (Sensitivity): 95.24%
•	F1-Score: 0.98

###  Confusion Matrix Results
Actual Class	Predicted Benign	Predicted Malignant
Benign	71	0
Malignant	2	41

The results demonstrate the model's ability to accurately identify malignant cases while maintaining a low number of misclassifications.

###  Potential Areas for Improvement
Future enhancements may include:
-	Explainable Artificial Intelligence (XAI) techniques such as SHAP and LIME
-	Class imbalance handling using SMOTE
-	Threshold optimization for improved sensitivity
-	Integration with medical imaging systems
-	Deployment within clinical decision-support environments

###  Potential Clinical Applications
The developed ANN model may support:
-	Clinical decision support for radiologists and physicians
-	Early cancer detection and risk assessment
-	Faster diagnostic workflows
-	Data-driven healthcare decision-making
-	Future integration with mammography, MRI, and ultrasound imaging systems

### Project Files
-	Dataset: Breast Cancer Wisconsin Diagnostic Dataset
-	Main Script: main.py
-	README: Project Documentation
