# Fetal Health Classification Using SVM

## Project Overview

This project applies supervised machine learning to classify fetal health into three categories using cardiotocography (CTG) features.

A Support Vector Machine (SVM) classifier with an RBF kernel is trained and evaluated using a structured machine-learning workflow. A Streamlit application provides an interactive interface for demonstrating predictions.

**This is an educational machine-learning project, not a clinically validated diagnostic system.**

## Objectives

* Understand and preprocess the dataset.
* Explore the distribution of fetal health classes.
* Train and evaluate an SVM classifier.
* Address class imbalance and tune model hyperparameters.
* Build an interactive application using Streamlit.

## Dataset

The dataset contains 21 input features derived from cardiotocography measurements and one target column, `fetal_health`.

The target classes are:

| Encoded Class | Description  |
| ------------- | ------------ |
| 1             | Normal       |
| 2             | Suspect      |
| 3             | Pathological |

The original dataset contained 2,126 records. After removing 13 duplicate records, 2,113 records remained.

The cleaned dataset contains no missing values.

## Machine Learning Workflow

1. Data loading and understanding
2. Data quality checks
3. Exploratory data analysis
4. Feature and target separation
5. Stratified train-test split
6. Feature scaling using `StandardScaler`
7. SVM model training
8. Class-imbalance handling
9. Hyperparameter tuning using `GridSearchCV`
10. Model evaluation
11. Model serialization
12. Interactive prediction application

## Model

* **Algorithm:** Support Vector Machine
* **Kernel:** RBF
* **Feature scaling:** StandardScaler
* **Hyperparameter tuning:** GridSearchCV
* **Cross-validation:** Stratified 5-fold cross-validation
* **Tuning metric:** Macro F1-score

The selected model was obtained through cross-validation on the training data.

## Results

The tuned SVM model achieved the following results on the held-out test set:

| Metric                      |            Result |
| --------------------------- | ----------------: |
| Accuracy                    |            92.20% |
| Macro F1-score              | Approximately 85% |
| Normal class F1-score       |              0.96 |
| Suspect class F1-score      |              0.74 |
| Pathological class F1-score |              0.86 |

The test set contained 423 records.

### Confusion Matrix

Rows represent actual classes, and columns represent predicted classes.

| Actual / Predicted | Normal | Suspect | Pathological |
| ------------------ | -----: | ------: | -----------: |
| Normal             |    317 |      11 |            2 |
| Suspect            |     12 |      42 |            4 |
| Pathological       |      1 |       3 |           31 |

### Interpretation

The tuned model achieved strong overall accuracy, but its performance differed across classes.

The Normal class had the strongest F1-score. The Suspect class was more difficult to classify, with a recall of approximately 72%.

Therefore, accuracy alone does not fully describe model performance. Per-class metrics and the confusion matrix are important when evaluating a multiclass classification model.

## Streamlit Application

The application allows users to enter the 21 feature values and obtain a model prediction.

Features include:

* Interactive numerical inputs
* Dataset-based input range validation
* Predicted class and encoded label
* Educational-use disclaimer

The application is intended to demonstrate model integration, not to support medical decisions.

## Project Structure

```text
Fetal_Health_Classification/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── raw/
├── models/
│   └── fetal_health_svm.pkl
└── notebooks/
    └── 01_fetal_health_classification.ipynb
```

## Installation

Clone the repository and navigate into the project directory:

```bash
git clone <YOUR_REPOSITORY_URL>
cd Fetal_Health_Classification
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Ensure the dataset and trained model are available in their expected directories.

## Run the Application

From the project root, execute:

```bash
streamlit run app.py
```

Streamlit will provide a local URL at which the application can be opened.

## Limitations

* This model is trained on a specific dataset and may not generalize to other populations or clinical settings.
* Overall accuracy does not guarantee reliable detection of every class.
* The input ranges are based on observed dataset values and do not establish clinical validity.
* The application has not been clinically validated.
* Predictions must not be used for diagnosis, treatment, or clinical decision-making.

## Future Improvements

* Add automated model evaluation and testing.
* Compare SVM with other classification algorithms.
* Improve minority-class performance.
* Add model explainability.
* Improve application usability and input guidance.
* Investigate generalization using appropriate external validation data.

## Disclaimer

This project is intended solely for educational and portfolio purposes. It is not a medical device or a substitute for professional medical assessment.
