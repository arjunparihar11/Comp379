import numpy as np
import pandas as pd

def process_data(filepath):
    #Convert CSV to DataFrame
    df = pd.read_csv(filepath)
    
    #Remove irrelevant features
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])
    
    #Convert Sex to numerical features
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Fill in missing values with their medians
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['SibSp'].fillna(df['SibSp'].median(), inplace=True)
    df['Parch'].fillna(df['Parch'].median(), inplace=True)
    
    return df

def rosenblatt_perceptron_algorithm(df, epochs, learning_rate, threshold):
    #Extract features and target
    X = df.drop(columns=['Survived']).values
    y = df['Survived'].values
    
    #Initialize weights to 0
    weights = np.zeros(X.shape[1])
    
    #Loop through ephoch number of times
    for epoch in range(epochs):
        for i in range(len(y)):
            #Compute weighted sum
            weighted_sum = np.dot(X[i], weights)
            
            #Predict based on the threshold
            prediction = 1 if weighted_sum >= threshold else 0
            
            #Update weights
            error = y[i] - prediction
            weights += learning_rate * error * X[i]
    
    return weights

def predict_survival(df, weights, threshold):
    #Extract features and target
    X = df.drop(columns=['Survived']).values
    y = df['Survived'].values
    
    #Compute the weighted sum
    weighted_sum = np.dot(X, weights)
    
    #Predict survival based on the threshold
    predictions = (weighted_sum >= threshold).astype(int)
    
    #Create a DataFrame to show the results
    results_df = pd.DataFrame({
        'Survived': y,
        'Predicted': predictions,
        'Weighted Sum': weighted_sum
    })
    
    return results_df

#Dataframe of training data
df = process_data('train.csv')

#Adjust Weights using rosenblatt perception algorithm (Loop 50 times)
weights = rosenblatt_perceptron_algorithm(df, 50, 0.001, 0)

#Make predictions and show results
results_df = predict_survival(df, weights, threshold=0)

#Print the first 5 rows
print(results_df)

#Calculate and print the total accuracy of predictions
accuracy = np.mean(results_df['Survived'] == results_df['Predicted'])
print(f"Total Accuracy: {accuracy:.2f}")