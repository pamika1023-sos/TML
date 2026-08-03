# Lab 01: KNN Classification on a Dataset of Your Choice

##  Objective
* Apply K-Nearest Neighbors (KNN) to classify a dataset of your choice.
* Compare the performance of different numbers of neighbors ($k$ values).

##  Contents & Steps
* **Dataset Selection:** Load and prepare the dataset (`healthcare-dataset-stroke-data.csv`).
* **Exploratory Data Analysis & Preprocessing:** Clean data, drop unnecessary columns (such as `id`), handle missing values, and encode categorical variables.
* **Standardization:** Scale input features using `StandardScaler` prior to training to ensure equal distance weighting.
* **Model Training:** Train KNN classification models using different $k$ values ($3, 5,$ and $7$).
* **Evaluation:** Evaluate each model's performance using accuracy and generate a detailed classification report.

##  Experimental Results & Output
* **Accuracy Scores for each $k$ value:**
  * $K = 3 \rightarrow$ Accuracy: **0.5988**
  * $K = 5 \rightarrow$ Accuracy: **0.6096**
  * $K = 7 \rightarrow$ Accuracy: **0.6106**

* **Best $k$ Value:** 
  * The optimal $k$ value based on test accuracy is **$k = 7$** with an accuracy of **0.6106**.

* **Classification Report ($K = 7$):**
  ```text
                precision    recall  f1-score   support

             0       0.62      0.57      0.60       515
             1       0.60      0.65      0.62       507

      accuracy                           0.61      1022
     macro avg       0.61      0.61      0.61      1022