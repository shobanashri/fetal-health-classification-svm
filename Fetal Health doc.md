##### **Fetal Health Classification using SVM**

\-----------------------------------------------------------



Work plan

\----------

| Step | Section                                        |  

| ---  | ---------------------------------------------- | 

|    1 | Create project folder and organize the dataset |  

|    2 | Load the dataset                               | 

|    3 | Data understanding                             | 

|    4 | EDA                                            | 

|    5 | Data cleaning                                  | 

|    6 | Feature-target separation and train-test split | 

|    7 | Scaling and baseline SVM                       | 

|    8 | Model evaluation                               | 

|    9 | Cross-validation and tuning                    | 

|   10 | Final model and documentation                  | 





Fetal-Health-SVM/

│

├── data/

│   └── raw/

│       └── fetal\_health.csv

│

├── notebooks/

│   └── 01\_fetal\_health\_classification.ipynb

│

├── models/

│

├── README.md

└── requirements.txt





Phase 1 : Data understanding

\-------------------------------

Goal: Understand the dataset before making changes.



* First and last five rows
* Number of rows and columns
* Column names
* Data types
* Missing values
* Duplicate records
* Target class distribution
* Summary statistics



Output: A clear understanding of the data's structure and quality.



***data set -> 2,126 row \& 22 col***

***i/p feature -> 21*** 

***target feature -> fetal\_health***

***22 col -> float64***

***missing value - 0***

***duplicate rows - 13 (in percent -> 0.61%)***

***rows remaining if duplicates are removed - 2113***

***complete row appear - 4 times***



***| Target value | Class        |***

***| -----------  | ------------ |***

***|            1 | Normal       |***

***|            2 | Suspect      |***

***|            3 | Pathological |***



***Class 1 (Normal): 1,655 observations (77.85%).***

***Class 2 (Suspect): 295 observations (13.88%).***

***Class 3 (Pathological): 176 observations (8.28%).***



***total -> 2,126*** 



***-> target variable is imbalance.***

***-> Normal observation dominate dataset***

***-> pathological observation form smallest class***



**-> Several features have many zero values**

**-> Some features may have skewed distributions -> The means are noticeably higher than the medians.**

**-> Some features have unusually large maximum values relative to their quartiles**

**-> The target variable is categorical, despite being stored as float64**

**1 — Normal**

**2 — Suspect**

**3 — Pathological**

**-> Several features have many zero or small values**

**-> Features have substantially different numerical scales**





**statistics**

**----------**

**count -> no. of non missing observations**

**mean -> avg value**

**std -> std deviation; measure spread**

**min -> min value**

**25% -> first quartile**

**50% -> median**

**75% -> third quartile**

**max -> max value**





**| Actual class     | Correct predictions | Total actual cases |              Recall |**

**| ---------------- | ------------------: | -----------------: | ------------------: |**

**| Normal (1)       |                 324 |                330 |              98.18% |**

**| Suspect (2)      |                  30 |                 58 |              51.72% |**

**| Pathological (3) |                  28 |                 35 |              80.00% |**

**| \*\*Total\*\*        |             \*\*382\*\* |            \*\*423\*\* | \*\*90.31% accuracy\*\* |**





**| Class                | Precision | Recall | F1-score | Interpretation                           |**

**| -------------------- | --------: | -----: | -------: | ---------------------------------------- |**

**| Normal               |    91.78% | 98.18% |   94.88% | Excellent recall                         |**

**| Suspect              |    75.00% | 51.72% |   61.22% | Needs improvement                        |**

**| Pathological         |    93.33% | 80.00% |   86.15% | Good performance                         |**

**| \*\*Overall accuracy\*\* |         — |      — |        — | \*\*90.31%\*\*                               |**

**| Macro average F1     |         — |      — |   80.75% | Shows performance across classes equally |**





***| Metric                            | Baseline SVM | Balanced SVM |***

***| --------------------------------- | -----------: | -----------: |***

***| Normal correctly classified       |          324 |          298 |***

***| Suspect correctly classified      |           30 |           52 |***

***| Pathological correctly classified |           28 |           31 |***

***| Total correct predictions         |          382 |          381 |***

***| Accuracy                          |       90.31% |       90.07% |***



***Suspect: Correctly classified cases increased from 30 to 52.***

***Pathological: Correctly classified cases increased from 28 to 31.***

***Normal: Correctly classified cases decreased from 324 to 298.***

***Pathological under-diagnoses: Your supplied confusion-matrix interpretation says zero Pathological cases were predicted as Normal. Please verify this directly against the matrix before recording it as a finding.***

***Medical reliability: We cannot conclude that the model is safe or reliable for medical use from these results alone.***



***Balanced SVM — Evaluation Findings***

***------------------------------------***

***Model: SVM with RBF kernel and class\_weight="balanced".***



***Results:***



***Accuracy: 90.07%***

***Macro F1-score: 84.09%***

***Normal recall: 90.30%***

***Suspect recall: 89.66%***

***Pathological recall: 88.57%***



***### Tuned SVM — Confusion Matrix Interpretation***



***The tuned SVM model correctly classified 390 of the 423 test samples, achieving a test accuracy of 92.20%.***



***\* \*\*Normal:\*\* 317 of 330 cases were correctly classified.***

***\* \*\*Suspect:\*\* 42 of 58 cases were correctly classified.***

***\* \*\*Pathological:\*\* 31 of 35 cases were correctly classified.***



***The model made 33 incorrect predictions. The Suspect class accounted for the largest number of misclassified samples, with 16 errors.***



***These results show that the tuned model performs well overall, but accuracy alone is insufficient to assess its ability to identify every class. Class-wise precision, recall, and F1-score must also be examined.***



***Hyperparameter Tuning***

***---------------------***

***GridSearchCV was used to optimize the balanced RBF SVM using five-fold stratified cross-validation and macro F1-score as the selection metric.***



***Best parameters:***



***C: 100***

***gamma: scale***

***class\_weight: balanced***

***Mean cross-validation Macro F1: 84.21%***

***Final Test Evaluation***



***The tuned SVM achieved 92.20% test accuracy, correctly classifying 390 of the 423 test samples. Its macro F1-score was approximately 85%.***



***The tuned model achieved better overall accuracy than the baseline and balanced SVM models. However, the balanced SVM achieved higher recall for the Suspect class. This highlights the trade-off between overall performance and identifying minority-class samples.***



***The model's results apply to this dataset and test split. They do not establish clinical reliability or suitability for medical diagnosis.***



Phase 2 — Exploratory Data Analysis (EDA)

\------------------------------------------

Goal: Discover patterns in the dataset.



* Distribution of Normal, Suspect, and Pathological classes
* Distributions of numerical features
* Correlation between features
* Feature relationships with fetal health
* Potential outliers and unusual values



Output: Key observations and conclusions we can explain in an interview.



Phase 3 — Data cleaning

\--------------------------

Goal: Prepare the dataset for modeling.



* Missing values
* Duplicate rows
* Invalid or unusual values
* Target labels



Output: A clean dataset ready for preprocessing.



Phase 4 — Feature and target separation

\----------------------------------------

Goal: Separate the inputs from the output.



X = data.drop("fetal\_health", axis=1)

y = data\["fetal\_health"]



X contains the 21 input features.

y contains the target class.



We'll implement this after confirming the dataset is ready.



Phase 5 — Train-test split

\----------------------------

Goal: Reserve unseen data to evaluate our model.



We'll start with an 80:20 train-test split.



We'll use stratify=y to preserve approximately the same class proportions in both sets.



Why? The three classes are imbalanced, so a random split without stratification could give us an unrepresentative class distribution.



Output: Training data and unseen test data.



Phase 6 — Feature scaling

\---------------------------

This is an essential step for SVM.



SVM is sensitive to feature scales. For example, one feature might have values around 0.003, while another has values around 150.



Without scaling, features with larger numerical ranges can disproportionately influence the model.



We'll use StandardScaler.



Importantly, we'll fit the scaler only on the training data. The test data will be transformed using that fitted scaler.



We'll use a scikit-learn Pipeline to keep preprocessing and modeling together and avoid data leakage.



Phase 7 — Build the SVM model

\------------------------------

Goal: Train our first classifier.



We'll begin with a simple baseline:



* Algorithm: SVC
* Kernel: rbf
* Scaling: StandardScaler
* Class handling: Initially evaluate the model with class\_weight="balanced" as a sensible option to test for the imbalanced target.



We'll compare the balanced model against an unweighted baseline rather than assuming either is better.



Why RBF? It can model nonlinear decision boundaries, making it a useful starting point for this dataset.



Phase 8 — Evaluate the model

\---------------------------------

Goal: Understand how well the model classifies each class.



We'll evaluate it on the held-out test data using:



| Metric           | Why we need it                                   |

| ---------------- | ------------------------------------------------ |

| Accuracy         | Overall proportion of correct predictions        |

| Precision        | How often predictions for a class are correct    |

| Recall           | How many actual cases of a class are identified  |

| F1-score         | Balance between precision and recall             |

| Confusion matrix | Shows which classes are confused with each other |



We'll pay particular attention to the Suspect and Pathological classes, because accuracy alone can hide poor performance on minority classes.



We won't aim for a particular accuracy number in advance. We'll report the actual results.



Phase 9 — Cross-validation and simple tuning

\----------------------------------------------

Once our baseline works, we'll use cross-validation on the training data and try a small number of SVM settings, such as:

* C
* gamma
* Kernel choice, if useful



We'll use a simple GridSearchCV search only after establishing the baseline.



The held-out test set will remain untouched until final evaluation.



Phase 10 — Final model and conclusions

\--------------------------------------

* Final model configuration
* Test-set evaluation metrics
* Confusion matrix
* Class-wise strengths and weaknesses
* Important limitations



We'll save the trained model only after we settle on the final pipeline.



Phase 11 — Documentation

\------------------------

Finally, we'll prepare:



* A clean notebook with explanations
* requirements.txt
* README.md
* A concise project conclusion



This will make the project easier to understand, reproduce, and showcase on GitHub.





Tech stack

\-----------

| Library        | Purpose                                 |

| -------------- | --------------------------------------- |

| `pandas`       | Load and inspect data                   |

| `numpy`        | Numerical operations                    |

| `matplotlib`   | Basic visualizations                    |

| `seaborn`      | Statistical plots                       |

| `scikit-learn` | Splitting, scaling, SVM, and evaluation |

| `joblib`       | Save and load the final model           |







