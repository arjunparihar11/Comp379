import numpy as np
from numpy.random import seed
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Perceptron:
    #Initialize perceptron parameters
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_func = self.step_function
        self.weights = None

    #Step activation function that returns 1 or 0
    def step_function(self, x):
        if x >= 0:
            return 1
        else:
            return 0
    
    #Training the perceptron on input data and labels
    #X: features, y: labels
    def fit(self, X, y):
        #The number of samples and features columns
        n_samples, n_features = X.shape
        #Initialize weights to zeros
        self.weights = np.zeros(n_features)
        #Iterate through the dataset
        for _ in range(self.epochs):
            for i, x_i in enumerate(X):
                #Linear combination of features and weights
                linear_output = np.dot(x_i, self.weights)
                #Step function
                y_hat = self.activation_func(linear_output)

                #Weight update rule
                update = self.learning_rate * (y[i] - y_hat)
                self.weights += update * x_i

    #Make predictions on new weights
    def predict(self, X):
        linear_output = np.dot(X, self.weights) 
        y_hat = self.activation_func(linear_output)
        return y_hat

#Textbook AdalineGD
class AdalineSGD(object):
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            np.random.seed(random_state)
        
    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]  # Directly return the shuffled arrays

    def _initialize_weights(self, m):
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, 0)
    
#1. Synthetic Linearly Separable Data
print(f"\n#1. Linearly Separable Data:")
#Pitch Speed and Spin Rate for fastball (1) or offspeed pitches (0)
baseball_pitches = np.array([[90, 2200], [85, 1800], [95, 2300], [100, 2500], [92, 2400],[80, 1600], [83, 1700], [88, 2000], [75, 1500], [82, 1650]])
pitch_labels = np.array([1, 0, 1, 1, 1, 0, 0, 0, 0, 0])
#Perceptron model
perceptron = Perceptron(learning_rate=0.01, epochs=100)
perceptron.fit(baseball_pitches, pitch_labels)
#Print final weights
#print("Final weights (Linearly Separable):", perceptron.weights)
# Measure accuracy
y_hat_linear = np.array([perceptron.predict(x) for x in baseball_pitches])
accuracy_linear = np.mean(y_hat_linear == pitch_labels)
accuracy_percentage = accuracy_linear * 100
print(f"Accuracy on Linearly Separable Dataset: {accuracy_percentage:.2f}%")

#2. Non-Linearly Separable Data
print(f"\n#2. Non-Linearly Separable Data:")
#Pitch Speed and Spin Rate for fastball (1) or offspeed pitches (0)
baseball_pitches_NL = np.array([[90, 2200], [85, 1800], [95, 2300], [100, 2500], [92, 2400],[80, 1600], [83, 1700], [88, 2000], [75, 1500], [82, 1650]])
pitch_labels_NL = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1])
# Perceptron model
perceptron.fit(baseball_pitches_NL, pitch_labels_NL)
# Print final weights
#print("Final weights (Non-Linearly Separable):", perceptron.weights)
# Measure accuracy
y_pred_nonlinear = np.array([perceptron.predict(x) for x in baseball_pitches_NL])
accuracypitch_labels_NL = np.mean(y_pred_nonlinear == pitch_labels_NL)*100
print(f"Accuracy on Non-Linearly Separable Dataset: {accuracypitch_labels_NL:.2f}%")

#3. Titanic Adaline Model
print(f"\n#3. Titanic Adaline Model:")
#Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
#Remove unnecessary columns
X = train_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'])
y = train_df['Survived'].to_numpy()
#Convert categorical variables
X = pd.get_dummies(X, drop_first=True)
#Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#Split into training (70%) and test (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
#Train the Adaline model
adaline = AdalineSGD(eta=0.01, n_iter=50, random_state=42)
adaline.fit(X_train, y_train)
#Evaluate on training data
train_predictions = adaline.predict(X_train)
train_accuracy = np.mean(train_predictions == y_train) * 100
print(f'Training Accuracy: {train_accuracy:.2f}%')
#Evaluate on test data
test_predictions = adaline.predict(X_test)
test_accuracy = np.mean(test_predictions == y_test) * 100
print(f'Test Accuracy: {test_accuracy:.2f}%')

#4. Most predictive features
print(f"\n#4. Most predictive features:")
def get_top_features(adaline, X):
    #Get the absolute values of the weights
    weights = adaline.w_[1:]
    feature_names = X.columns
    #Create a DataFrame with feature names and their corresponding weights
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Weight': weights})
    #Sort the DataFrame by absolute weight values in descending order
    feature_importance['Abs_Weight'] = feature_importance['Weight'].abs()
    top_features = feature_importance.sort_values(by='Abs_Weight', ascending=False).head()
    return top_features[['Feature', 'Weight']]
top_features = get_top_features(adaline, X)
print("\nTop 5 Most Predictive Features:")
print(top_features)

#5. Baseline Model
print(f"\n#5. Baseline Model")
#Create a random baseline model
class Baseline:
    def __init__(self, n_features):
        #Initialize with random weights
        self.weights = np.random.rand(n_features)
    def predict(self, X):
        #Make predictions based on random weights
        linear_output = np.dot(X, self.weights)
        return np.where(linear_output >= 0.5, 1, 0)
#Instantiate the baseline model
baseline_model = Baseline(n_features=X.shape[1])
#Train the baseline model using random weights
baseline_predictions_train = baseline_model.predict(X_train)
baseline_accuracy_train = np.mean(baseline_predictions_train == y_train) * 100
print(f'Baseline Training Accuracy: {baseline_accuracy_train:.2f}%')
baseline_predictions_test = baseline_model.predict(X_test)
baseline_accuracy_test = np.mean(baseline_predictions_test == y_test) * 100
print(f'Baseline Test Accuracy: {baseline_accuracy_test:.2f}%')