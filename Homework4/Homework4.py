import numpy as np
import pandas as pd
from pybaseball import statcast_pitcher, playerid_lookup
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

#1. Randomly split the data into training (80%) and test (20%) sets.
def collect_data(player_id, start_date='2024-03-28', end_date='2024-09-30'):
    data = statcast_pitcher(start_date, end_date, player_id)
    selected_features = [
        'pitch_type',                       #Target variable
        'inning',                           #Game context
        'balls', 'strikes',                 #Count context
        'outs_when_up',                     #Game context
        'release_speed',                    #Speed of the pitch
        'release_spin_rate',                #Spin rate of the pitch
        'release_pos_x', 'release_pos_z',   #Release position of the pitch
        'pfx_x', 'pfx_z',                   #Horizontal and vertical movement
        'plate_x', 'plate_z',               #Final location of the pitch
        'vx0', 'vy0', 'vz0',                #Velocity components of the pitch
        'ax', 'ay', 'az'                    #Acceleration components of the pitch
    ]
    data = data[selected_features].dropna()  #Drop rows with missing values
    return data

def split_data(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    return pd.DataFrame(train, columns=data.columns), pd.DataFrame(test, columns=data.columns)

#2. Train and evaluate a classifier using n-fold cross validation from scratch.
def n_fold_cv(model, data, target, n_folds=5):
    data_split = np.array_split(data, n_folds)
    scores = []
    for i in range(n_folds):
        validation = data_split[i]
        training = pd.concat([data_split[j] for j in range(n_folds) if j != i], ignore_index=True)
        X_train, y_train = training.drop(columns=[target]), training[target]
        X_val, y_val = validation.drop(columns=[target]), validation[target]
        model.fit(X_train, y_train)
        scores.append(model.score(X_val, y_val))
    return np.mean(scores)

#3. Implement your own grid search from scratch with a search over at least two hyper-parameters.
def custom_grid_search(model_class, param_grid, data, target, n_folds=5):
    best_score = -float('inf')
    best_params = {}
    for params in param_grid:
        for C in params['C']:
            for penalty in params['penalty']:
                model = model_class(C=C, penalty=penalty, solver='liblinear', max_iter=500)
                score = n_fold_cv(model, data, target, n_folds=n_folds)
                if score > best_score:
                    best_score = score
                    best_params = {'C': C, 'penalty': penalty}
    return best_params, best_score

#4. Evaluate the best model that you identified and report its performance on the test set. C
def get_player_id(first_name, last_name):
    player_info = playerid_lookup(last_name, first_name)
    player_id = player_info.loc[player_info.index[0], 'key_mlbam']
    return player_id

def main():
    player_id = get_player_id('Zack', 'Wheeler')
    data = collect_data(player_id)
    data['pitch_type'] = data['pitch_type'].astype('category').cat.codes
    train_data, test_data = split_data(data)
    param_grid = [
        {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l2']},
        {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1']}
    ]
    best_params, _ = custom_grid_search(LogisticRegression, param_grid, train_data, 'pitch_type', n_folds=5)
    best_model = LogisticRegression(**best_params, solver='liblinear', max_iter=500)
    best_model.fit(train_data.drop(columns=['pitch_type']), train_data['pitch_type'])
    predictions = best_model.predict(test_data.drop(columns=['pitch_type']))
    accuracy = accuracy_score(test_data['pitch_type'], predictions)
    f1 = f1_score(test_data['pitch_type'], predictions, average='weighted')
    print("Best Parameters:", best_params)
    print("Test Set Accuracy:", accuracy)
    print("Test Set F1 Score:", f1)

if __name__ == "__main__":
    main()
