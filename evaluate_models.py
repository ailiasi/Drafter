import models
import pandas as pd
from data_processing import binary_encode
from sklearn.model_selection import train_test_split, ParameterGrid

def read_and_split_data(filename, nrows = None):
    data = pd.read_csv(filename, nrows = nrows,
                       usecols = ["hero" + str(i) for i in range(1,10)] +["map0","map1","mode0","mode1","winner"])
    data = data.apply(binary_encode, axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-2], data.iloc[:,-2:])
    return X_train, X_test, y_train, y_test



def fit_simple_model(X_train, X_test, y_train, y_test, 
                     input_shape=(260,), 
                     output_nodes=2, 
                     num_hid_layers=1, 
                     num_hid_nodes=200, 
                     dropout=0.5, 
                     epochs=10, 
                     batch_size=32,
                     verbose=0):
    model = models.simple_model(input_shape, output_nodes, num_hid_layers, num_hid_nodes, dropout)
    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), 
                        batch_size = batch_size, epochs = epochs, verbose = verbose)
    return model, history

def simple_model_grid_search(result_filename):
    input_shape, output_nodes, epochs, batch_size = [(260,)], [2], [10], [32]
    
    X_train, X_test, y_train, y_test = read_and_split_data("data/processed/teams_20181001-20190123_encoded.csv", nrows = 100000)
    
    hid_layers = [2]
    hid_nodes = [100,200,300]
    dropout = [0.5]
    param_grid = dict(input_shape = input_shape, output_nodes = output_nodes,
                      num_hid_layers = hid_layers, num_hid_nodes = hid_nodes, 
                      dropout = dropout, epochs = epochs, batch_size = batch_size)
    
    for params in ParameterGrid(param_grid):
        print(params)
        model, history = fit_simple_model(X_train, X_test, y_train, y_test, **params)
        with open(result_filename, 'a') as f:
            f.write(str(params) + "\n")
            for key in history.history.keys():
                f.write(key + "," + ",".join(map(str, history.history[key])) + "\n")
                
                
    