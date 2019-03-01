from . import models
import keras
import pandas as pd
from data_processing import binary_encode
from sklearn.model_selection import train_test_split, ParameterGrid

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, 
                              patience=2, verbose=0, 
                              mode='auto', baseline=None, restore_best_weights=True)

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
                     regularization=0,
                     epochs=10, 
                     batch_size=32,
                     verbose=0):
    model = models.simple_model(input_shape, output_nodes, num_hid_layers, num_hid_nodes, dropout, regularization)
    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), 
                        batch_size = batch_size, epochs = epochs, verbose = verbose, 
                        callbacks = [early_stopping])
    return model, history

def fit_siamese_model(X_train, X_test, y_train, y_test,
                      input1_shape=(130,), input2_shape=(130,),
                      output_nodes=2,
                      num_siam_layers=1, num_siam_nodes=200, 
                      num_hid_layers=1, num_hid_nodes=200,
                      dropout=0,
                      regularization=0,
                      epochs=10, 
                      batch_size=32,
                      verbose=0):
    model = models.siamese_model(input1_shape, input2_shape,
                                 output_nodes,
                                 num_siam_layers, num_siam_nodes, 
                                 num_hid_layers, num_hid_nodes,
                                 dropout, regularization)
    
    history = model.fit([X_train[0],X_train[1]], y_train, validation_data = ([X_test[0],X_test[1]], y_test),
                        batch_size = batch_size, epochs = epochs, verbose = verbose, 
                        callbacks = [early_stopping])
    return model, history

def simple_model_grid_search(result_filename):
    input_shape, output_nodes, epochs, batch_size = [(260,)], [2], [10], [1000]
    
    print("reading data...")
    X_train, X_test, y_train, y_test = read_and_split_data("data/processed/teams_20181001-20190123_encoded.csv", nrows = 100000)
    
    hid_layers = [2]
    hid_nodes = [200]
    dropout = [0.5]
    param_grid = dict(input_shape = input_shape, output_nodes = output_nodes,
                      num_hid_layers = hid_layers, num_hid_nodes = hid_nodes, 
                      dropout = dropout, epochs = epochs, batch_size = batch_size)
    
    
    for params in ParameterGrid(param_grid):
        print(params)
        model, history, loss_history = fit_simple_model(X_train, X_test, y_train, y_test, **params)
        with open(result_filename, 'a') as f:
            f.write(str(params) + "\n")
            for key in history.history.keys():
                f.write(key + "," + ",".join(map(str, history.history[key])) + "\n")
    return loss_history
                
def siamese_model_grid_search(result_filename):
    input1_shape, input2_shape, output_nodes, epochs, batch_size = [(130,)],[(130,)], [2], [50], [32]
    
    X_train, X_test, y_train, y_test = read_and_split_data("data/processed/teams_20181001-20190123_encoded.csv")
    
    siam_layers = [1]
    siam_nodes = [200]
    hid_layers = [1]
    hid_nodes = [200]
    dropout = [0]
    param_grid = dict(input1_shape = input1_shape, input2_shape = input2_shape, 
                      output_nodes = output_nodes,
                      num_siam_layers = siam_layers, num_siam_nodes = siam_nodes,
                      num_hid_layers = hid_layers, num_hid_nodes = hid_nodes, 
                      dropout = dropout, epochs = epochs, batch_size = batch_size)
    
    for params in ParameterGrid(param_grid):
        print(params)
        model, history = fit_siamese_model([X_train.iloc[:,:130],X_train.iloc[:,130:]], 
                                           [X_test.iloc[:,:130],X_test.iloc[:,130:]], 
                                           y_train, y_test, **params)
        with open(result_filename, 'a') as f:
            f.write(str(params) + "\n")
            for key in history.history.keys():
                f.write(key + "," + ",".join(map(str, history.history[key])) + "\n")                    