## 📊 Datasets Overview

For this project, we selected four real-world datasets that strictly match the requirements: they correspond to binary classification problems and contain a relatively large number of numerical features to ensure sufficient model complexity for evaluating the logistic regression model with the FISTA algorithm.

### 1. Spambase Dataset
* **Description:** This dataset was created to classify emails as either spam or non-spam. It is a classic benchmark for binary classification tasks in natural language processing.
* **Features:** Contains 57 continuous numerical variables representing the frequency of specific words (e.g., "free", "money") and characters (e.g., "!", "$") in the email text, as well as measurements of capital letter sequences.
* **Target Variable:** Binary (1 = Spam, 0 = Non-spam).
* **Source:** [UCI Machine Learning Repository - Spambase](https://archive.ics.uci.edu/dataset/94/spambase)

### 2. Connectionist Bench (Sonar, Mines vs. Rocks) Dataset
* **Description:** This dataset involves the task of training a network to discriminate between sonar signals bounced off a metal cylinder (representing a naval mine) and those bounced off a roughly cylindrical rock.
* **Features:** Contains 60 numerical features in the range of 0.0 to 1.0. Each number represents the energy within a particular frequency band, integrated over a certain period of time.
* **Target Variable:** Binary (M = Mine, R = Rock). *Note: Labels will be mapped to 1 and 0 for the logistic regression.*
* **Source:** [UCI Machine Learning Repository - Sonar](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks)

### 3. Breast Cancer Wisconsin (Diagnostic) Dataset
* **Description:** A highly recognized dataset in medical machine learning used to predict whether a breast mass is malignant or benign based on digitized images of a fine needle aspirate (FNA).
* **Features:** Contains 30 continuous numerical features computed from the cell nuclei present in the image. These include measurements like radius, texture, perimeter, area, smoothness, and compactness.
* **Target Variable:** Binary (M = Malignant, B = Benign). *Note: Labels will be mapped to 1 and 0.*
* **Source:** [Kaggle / UCI Machine Learning Repository - Breast Cancer](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

### 4. Phishing Websites Dataset
* **Description:** This dataset is designed to identify phishing websites based on their structural characteristics and URL properties, which is a crucial task in cybersecurity.
* **Features:** Contains 30 features extracted from websites, such as URL length, having an '@' symbol, prefix/suffix in the domain, and SSL final state. While conceptually categorical, the features are encoded numerically (e.g., -1, 0, 1), making them suitable for this mathematical optimization context.
* **Target Variable:** Binary (1 = Legitimate, -1 = Phishing). *Note: The negative class will be remapped to 0 for standard binary classification formulation.*
* **Source:** [UCI Machine Learning Repository - Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing+websites)