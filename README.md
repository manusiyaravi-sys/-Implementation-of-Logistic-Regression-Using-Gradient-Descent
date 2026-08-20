# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the required libraries and load the dataset.

2. Split the dataset into training data and testing data.

3. Create and train the Logistic Regression model using the training data.

4. Predict the output values for the test data using the trained model.

5. Evaluate the model performance using accuracy score, classification report, and confusion matrix.

```
## Program:
```

/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Manusiya R
RegisterNumber: 212225240083
*/
# Logistic Regression - Accuracy, Classification Report & Confusion Matrix

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create Logistic Regression Model
model = LogisticRegression(max_iter=5000)

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification Report
print("\nclassification report")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

```

## Output:
<img width="946" height="373" alt="image" src="https://github.com/user-attachments/assets/c4c08231-a810-42d7-ba49-586f75866a4d" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

