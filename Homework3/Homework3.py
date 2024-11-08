import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

#Heart Disease UCI Data Set to Predict if Heart Disease is present
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
df = pd.read_csv(url, names=column_names)
#Replace ? with NaN and drop missing values
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)
#Convert categorical features to numeric
df = pd.get_dummies(df, columns=["cp", "restecg", "slope", "thal"], drop_first=True)
#Split features (X) and target (y)
X = df.drop(columns=['target'])
y = df['target']
#Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#Split the dataset: 70% training, 15% development, 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=11, stratify=y)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=11, stratify=y_temp)
print(f"Training set: {len(X_train)} samples")
print(f"Development set: {len(X_dev)} samples")
print(f"Test set: {len(X_test)} samples")

#1. Train with SVM with default hyperparameters
print("\n#1. Train with SVM with default hyperparameters\n")
svm_clf = SVC(random_state=42)
svm_clf.fit(X_train, y_train)
#Predict on the development set
y_dev_pred = svm_clf.predict(X_dev)
#Compute accuracy
accuracy = np.mean(y_dev_pred == y_dev)
print(f"SVM Accuracy: {accuracy:.4f}")

#2. Explore the classifier hyperparameters
print("\n#2. Explore the classifier hyperparameters\n")
#List of C values to try
C_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
#Store the best model and its performance
best_C = None
best_accuracy = 0
best_model = None
for C in C_values:
    #Train SVM with specific C value
    svm_clf = SVC(C=C, random_state=42)
    svm_clf.fit(X_train, y_train)
    #Predict on the development set
    y_dev_pred = svm_clf.predict(X_dev)
    #Compute accuracy
    accuracy = np.mean(y_dev_pred == y_dev)
    print(f"C={C}, Accuracy={accuracy:.4f}")
    #Keep track of the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_C = C
        best_model = svm_clf
#Best model information
print(f"\nBest C value: {best_C}, Best Accuracy: {best_accuracy:.4f}")

#3. Implement your own KNN from scratch
print("\n#3. Implement your own KNN from scratch\n")
class KNN:
    def __init__(self, k=3):
        self.k = k
    def fit(self, X_train, y_train):
        self.X_train = X_train
        #Convert y_train to a NumPy array
        self.y_train = np.asarray(y_train, dtype=int)
    def predict(self, X_test):
        #Ensure that X_test has the correct number of features
        if len(X_test.shape) != 2 or X_test.shape[1] != self.X_train.shape[1]:
            raise ValueError(f"X_test should have {self.X_train.shape[1]} features, but got {X_test.shape[1]}")
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)
    def _predict(self, x):
        #Compute Euclidean distances
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        #Sort by distance
        k_indices = np.argsort(distances)[:self.k]
        #Extract the labels of the k nearest neighbor
        k_nearest_labels = self.y_train[k_indices]  # Access labels with valid indices
        #Find the most frequent class label
        most_common_label = np.argmax(np.bincount(k_nearest_labels))
        return most_common_label
#Example usage:
knn = KNN(k=3)
knn.fit(X_train, y_train)
#Predict on the development set
y_dev_pred = knn.predict(X_dev)
accuracy = np.mean(y_dev_pred == y_dev)
print(f"KNN Accuracy: {accuracy:.4f}")

#4. Establish the performance of a baseline system
print("\n#4. Establish the performance of a baseline system\n")
#Initialize and fit the Dummy Classifier with stratified strategy
dummy_stratified = DummyClassifier(strategy='stratified', random_state=42)
dummy_stratified.fit(X_train, y_train)
#Predict on the development set
y_dev_dummy_stratified = dummy_stratified.predict(X_dev)
#Compute accuracy
dummy_accuracy_stratified = np.mean(y_dev_dummy_stratified == y_dev)
print(f"Dummy Classifier (Stratified) Accuracy: {dummy_accuracy_stratified:.4f}")
#Initialize and fit the Dummy Classifier with most_frequent strategy
dummy_most_frequent = DummyClassifier(strategy='most_frequent')
dummy_most_frequent.fit(X_train, y_train)
#Predict on the development set
y_dev_dummy_most_frequent = dummy_most_frequent.predict(X_dev)
#Compute accuracy
dummy_accuracy_most_frequent = np.mean(y_dev_dummy_most_frequent == y_dev)
print(f"Dummy Classifier (Most Frequent) Accuracy: {dummy_accuracy_most_frequent:.4f}")

#5. Comparing Best KNN Model to Dummy Classifier
print("\n#5. Comparing Best KNN Model to Dummy Classifier\n")
#Predict on the development set using the KNN model
y_dev_knn_pred = knn.predict(X_dev)
#Compute KNN accuracy
knn_accuracy = np.mean(y_dev_knn_pred == y_dev)
print(f"KNN Accuracy: {knn_accuracy:.4f}")
#Compare the accuracies
print("\nComparison of Accuracies:")
print(f"KNN Accuracy: {knn_accuracy:.4f}")
print(f"Dummy Classifier (Stratified) Accuracy: {dummy_accuracy_stratified:.4f}")
print(f"Dummy Classifier (Most Frequent) Accuracy: {dummy_accuracy_most_frequent:.4f}")