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





statistics

\----------

count -> no. of non missing observations

mean -> avg value

std -> std deviation; measure spread

min -> min value

25% -> first quartile

50% -> median

75% -> third quartile

max -> max value

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







