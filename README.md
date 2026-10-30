# 📩 SMS Spam Detection Using Machine Learning

A Machine Learning-based SMS Spam Detection System that automatically classifies text messages into two categories: **Spam** and **Ham (Not Spam)** using Natural Language Processing (NLP) and the Multinomial Naive Bayes algorithm.

The main objective of this project is to identify unwanted, promotional, and potentially fraudulent messages while correctly recognizing genuine conversations.

---

## 🚀 Project Overview

SMS spam messages have become a common problem in digital communication. These messages may contain fake offers, lottery notifications, suspicious links, fraudulent requests, and unwanted advertisements.

This project uses Machine Learning to analyze message content and predict whether a given message is spam or ham.

The system follows a supervised learning approach. It learns patterns from a labeled dataset and uses those patterns to classify new, unseen messages.

### 🎯 Project Objectives

- Detect spam and legitimate SMS messages automatically.
- Apply Natural Language Processing (NLP) techniques.
- Understand text preprocessing and feature extraction.
- Train a classification model using Multinomial Naive Bayes.
- Evaluate model performance using standard classification metrics.
- Build a reusable spam detection system.

---

## 🧠 Machine Learning Algorithm

**Multinomial Naive Bayes**

Multinomial Naive Bayes is a probabilistic classification algorithm commonly used for text classification tasks.

It calculates the probability of a message belonging to a particular class based on the words present in that message.

The algorithm uses Bayes' Theorem:

\[
P(C|X)=\frac{P(X|C)\times P(C)}{P(X)}
\]

Where:

- \(P(C|X)\): Probability of a class given the message.
- \(P(X|C)\): Probability of the message features given the class.
- \(P(C)\): Prior probability of the class.
- \(P(X)\): Probability of the message features.

The model predicts whether a message belongs to the Spam or Ham category.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Dataset handling and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning model |
| NLP | Text processing |
| CountVectorizer / TF-IDF | Text feature extraction |
| Matplotlib | Data visualization |
| Seaborn | Confusion matrix visualization |
| Joblib | Saving and loading trained models |
| Streamlit (Optional) | Web application interface |
| Git & GitHub | Version control and project hosting |

---

## 📂 Project Structure

```text
SMS-Spam-Detection/
│
├── dataset/
│   └── spam.csv
│
├── model/
│   ├── spam_model.pkl
│   └── vectorizer.pkl
│
├── notebooks/
│   └── spam_detection.ipynb
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

*Note: The directory structure may vary depending on your implementation.*

---

## 📊 Dataset Description

The project uses a labeled SMS dataset containing two categories:

| Label | Category | Description |
|---|---|---|
| 0 | Ham | Genuine and normal messages |
| 1 | Spam | Unwanted or suspicious messages |

### Example Dataset

| SMS Message | Label |
|---|---|
| Congratulations! You won a free iPhone. | Spam |
| Hello, how are you? | Ham |
| Claim your free prize now! | Spam |
| I will call you later. | Ham |
| You have won a lottery. Click here! | Spam |
| Please send me the assignment. | Ham |

### Dataset Source

You can use the publicly available SMS Spam Collection dataset:

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

---

## ⚙️ Project Workflow

The project follows these major steps:

**1. Data Collection**

Collect labeled SMS messages containing spam and ham examples.

**2. Data Preprocessing**

- Remove unnecessary columns.
- Handle missing values.
- Remove duplicate messages.
- Clean and normalize text.
- Prepare the target labels.

**3. Exploratory Data Analysis (EDA)**

Analyze the dataset to understand:

- Number of spam messages.
- Number of ham messages.
- Distribution of message categories.
- Common words in spam and ham messages.

**4. Feature Extraction**

Convert text messages into numerical features using CountVectorizer or TF-IDF.

Machine Learning algorithms cannot directly understand raw text, so feature extraction converts text into a numerical representation.

**5. Model Training**

Train the Multinomial Naive Bayes classifier using the training dataset.

**6. Model Evaluation**

Evaluate the trained model using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

**7. Prediction**

Use the trained model to classify new messages as Spam or Ham.

---

## 💻 Installation and Setup

### Prerequisites

Make sure you have installed:

- Python 3.9 or above
- pip
- Git
- VS Code (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

### Step 2: Navigate to the Project Directory

```bash
cd YOUR_REPOSITORY
```

### Step 3: Create a Virtual Environment

```bash
python3 -m venv venv
```

### Step 4: Activate the Virtual Environment

For Ubuntu/Linux:

```bash
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Project

If using a Python training script:

```bash
python train.py
```

If using Streamlit:

```bash
streamlit run app.py
```

---

## 📈 Model Evaluation

The performance of the model is evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall correct predictions |
| Precision | Measures how many predicted spam messages are actually spam |
| Recall | Measures how many actual spam messages are detected |
| F1-Score | Harmonic mean of precision and recall |
| Confusion Matrix | Shows correct and incorrect classifications |

### Confusion Matrix

A confusion matrix contains four important values:

- True Positive (TP): Spam correctly classified as spam.
- True Negative (TN): Ham correctly classified as ham.
- False Positive (FP): Ham incorrectly classified as spam.
- False Negative (FN): Spam incorrectly classified as ham.

**Important:** Actual evaluation scores should be added here after running the model on your test dataset. Do not report assumed accuracy values.

---

## 🧪 Sample Predictions

| Input Message | Expected Output |
|---|---|
| Congratulations! You won a lottery! | Spam |
| Claim your free gift now! | Spam |
| Earn money from home. Register today! | Spam |
| Hello, how are you? | Ham |
| I will call you later. | Ham |
| Please send me the assignment. | Ham |

---

## 🔍 Example Python Prediction

```python
message = ["Congratulations! You won a free prize!"]

message_vector = vectorizer.transform(message)

prediction = model.predict(message_vector)

if prediction[0] == 1:
    print("Spam Message")
else:
    print("Ham Message")
```

---

## 🔮 Future Improvements

The project can be extended with the following features:

- Develop a user-friendly Streamlit web application.
- Add real-time SMS spam detection.
- Improve model performance using additional datasets.
- Compare Multinomial Naive Bayes with Logistic Regression, Random Forest, and SVM.
- Add visualization of prediction results.
- Implement a spam probability score.
- Explore deep learning models such as LSTM and BERT.
- Deploy the application on a cloud platform.

---

## 🎓 Learning Outcomes

Through this project, I learned:

- Fundamentals of supervised Machine Learning.
- Text preprocessing using NLP.
- Feature extraction using vectorization techniques.
- Working with the Multinomial Naive Bayes algorithm.
- Model training and evaluation.
- Understanding accuracy, precision, recall, and F1-score.
- Saving and loading trained Machine Learning models.
- Managing projects using Git and GitHub.

---

## 👨‍💻 Author

**Abhijit Dutta**

BSc Computer Science Student | Machine Learning Enthusiast | AI & Technology Explorer

- GitHub: https://github.com/abhiduttaa
- Portfolio: https://abhiduttaa.github.io/portfoliowebsiteusinghtml/

---

## 📜 License

This project is licensed under the MIT License. You are free to use, modify, and distribute this project according to the license terms.

---

⭐ **If you find this project useful, consider giving it a star on GitHub!**

*Built with Python, Machine Learning, and a passion for Artificial Intelligence.*
