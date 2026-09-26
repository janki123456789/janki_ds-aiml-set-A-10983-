# 📊 Data Science & AI/ML Practical Exam – Set A

## 🎯 Project Overview

This project is a complete **Data Science & AI/ML practical project** based on campaign response prediction and audience segmentation.

The project covers:

* 📈 Maths & Advanced Statistics
* 🧹 Data Preprocessing & Feature Engineering
* 🤖 Supervised Learning
* 🔵 Unsupervised Learning
* 🧠 Deep Learning using ANN
* 📊 Data Visualization
* 📁 Reproducible Machine Learning Workflow

The dataset contains synthetic campaign-response observations with numerical features and an operational group variable.

---

## 👩‍💻 Student Information

| Details          | Information                    |
| ---------------- | ------------------------------ |
| **Student Name** | Janki Dholariya                |
| **Exam Set**     | Set A                          |
| **Project Type** | Data Science & AI/ML Practical |
| **Language**     | Python                         |
| **Notebook**     | `SET-A.ipynb`                   |

---

# 🎯 Objective

The main objectives of this project are:

1. Analyze campaign-related numerical data using statistical methods.
2. Perform data cleaning and feature engineering.
3. Predict campaign response using Logistic Regression.
4. Compare the classifier with a majority-class baseline.
5. Identify audience segments using K-Means clustering.
6. Build a Feed-Forward Artificial Neural Network.
7. Compare ANN performance with Logistic Regression.
8. Visualize important statistical and machine learning results.

---

# 📂 Dataset

The dataset is a **synthetically generated dataset** supplied for the practical exam.

### Dataset File

- [set_d](data/raw/set_d.csv)


### Dataset Information

| Feature      | Description                |
| ------------ | -------------------------- |
| `record_id`  | Unique record identifier   |
| `visits`     | Synthetic visits index     |
| `recency`    | Synthetic recency index    |
| `engagement` | Synthetic engagement index |
| `spend`      | Synthetic spending index   |
| `group`      | Operational group: G1 / G2 |
| `response`   | Binary target: 0 or 1      |

`record_id` is used only as an identifier and is excluded from model training.

---

# 📊 Dataset Generation

The supplied generator creates:

* **305 total rows**
* **300 unique records**
* **5 exact duplicate rows**
* Missing values in selected numerical columns

The raw dataset is preserved unchanged.

---

# 📌 Train / Validation / Test Split

The dataset was split using:

```python
random_state = 42
```

The final partition sizes were:

| Partition      | Records |
| -------------- | ------: |
| Fit / Training |     192 |
| Validation     |      48 |
| Test           |      60 |
| **Total**      | **300** |

The partitions were verified to be disjoint.

---

# 📈 Task 1 – Maths & Advanced Statistics

## Descriptive Statistics

The project calculates:

* Observed sample size
* Mean
* Median
* Sample standard deviation
* Histogram

---

# 🤖 Task 2 – Data Preprocessing & Feature Engineering

The preprocessing pipeline includes:

```text
Raw Dataset
     ↓
Duplicate Removal
     ↓
Train / Validation / Test Split
     ↓
Median Imputation
     ↓
Feature Engineering
     ↓
One-Hot Encoding
     ↓
Standard Scaling
     ↓
Machine Learning Models
```

T

# 🎯 Task 3 – Supervised Learning
```

### Test Results

| Model               | Accuracy | Precision | Recall |     F1 |
| ------------------- | -------: | --------: | -----: | -----: |
| Dummy Classifier    |   0.5833 |         — |      — |      — |
| Logistic Regression |   0.8167 |    0.8333 | 0.8571 | 0.8451 |

The Logistic Regression model achieved an F1-score of approximately **0.8451** on the untouched test set.
```
---

# 🔵 Task 4 – Unsupervised Learning

## K-Means Clustering

K-Means was evaluated for:

```text
k = 2
k = 3
k = 4
```
Parameters:

```python
n_init = 10
random_state = 42
```

### Cluster Profiles

| Cluster | Visits | Recency | Engagement | Spend | Engineered Feature |
| ------: | -----: | ------: | ---------: | ----: | -----------------: |
|       0 |  49.19 |   47.15 |      51.39 | 55.69 |             0.8373 |
|       1 |  49.86 |   55.72 |      50.38 | 42.28 |             1.3052 |

Cluster IDs are labels only and do not represent target classes.

---

# 🧠 Task 5 – Deep Learning / ANN

A Feed-Forward Artificial Neural Network was implemented using TensorFlow/Keras.

### Configuration

```text
Optimizer: Adam
Learning Rate: 0.001
Loss: Binary Cross-Entropy
Batch Size: 16
Maximum Epochs: 50
Early Stopping Patience: 5
Random Seed: 42
Trainable Parameters: 273
```

The sigmoid output is suitable for binary classification, while binary cross-entropy is used as the loss function.
---

# 📁 Project Structure

```text
ds-aiml-set-a/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── raw/
│       └── set_d.csv
│
├── src/
│   └── generate_data.py
│
├── notebooks/
│   └── exam.ipynb
│
├── outputs/
│   ├── splits.csv
│   ├── statistics_summary.csv
│   ├── metrics.csv
│   ├── predictions.csv
│   ├── cluster_profiles.csv
│   │
│   └── figures/
│       ├── dataset_distribution.png
│       ├── histogram.png
│       ├── logistic_confusion_matrix.png
│       ├── kmeans_clusters.png
│       ├── ann_loss_curve.png
│       └── model_comparison.png
│
└── models/
    └── trained_model_files
```

---

# 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* SciPy
* Scikit-learn
* Matplotlib
* TensorFlow
* Keras
* Jupyter Notebook
* Git & GitHub

---

# 📊 Important Findings

### Finding 1 – Supervised Learning

Logistic Regression achieved:

```text
Accuracy = 81.67%
Precision = 83.33%
Recall = 85.71%
F1 Score = 84.51%
```
---

# 🎥 Project Explanation Video

**Video Duration:** 5–10 minutes

🔗 **Video Link:**
`PASTE_YOUR_GOOGLE_DRIVE_OR_YOUTUBE_LINK_HERE`

The video should demonstrate:

* Dataset generation and cleaning
* Statistical analysis
* Data preprocessing
* Logistic Regression
* Confusion matrix
* K-Means clustering
* ANN architecture
* ANN loss curve
* Model comparison
* GitHub repository

---

# ✅ Reproducibility

The project uses:

```text
Random State = 42
```

The same test IDs are used for the final comparison between Logistic Regression and ANN.

Preprocessing transformations are fitted only on the fit/training data.

---

# 👩‍💻 Author

**Janki Dholariya**

Data Science & AI/ML Practical Project – Set A

---
