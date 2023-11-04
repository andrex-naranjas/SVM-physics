import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 1: Load the data from the CSV file
data = pd.read_csv("spherical_data.csv")

# Assuming the target column is named 'class', and you have feature columns named 'feature1', 'feature2', etc.
X = data.drop('class', axis=1)
y = data['class']

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train the SVM classifier
svm_classifier = SVC(kernel='poly', degree=2)  # You can choose different kernels such as 'linear', 'rbf', 'poly', etc.
#svm_classifier = SVC(kernel='linear')  # You can choose different kernels such as 'linear', 'rbf', 'poly', etc.
svm_classifier.fit(X_train, y_train) # fit -> teach

# Step 4: Make predictions on the test set
y_pred = svm_classifier.predict(X_test)
print(y_pred)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
