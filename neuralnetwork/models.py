import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers


def siamese_model(input1_shape, input2_shape,
                  output_nodes,
                  num_siam_layers, num_siam_nodes, 
                  num_hid_layers, num_hid_nodes,
                  dropout=0, regularization =0):
    # TODO: add dropout
    
    input_a = Input(shape = input1_shape, name = "input_a")
    input_b = Input(shape = input2_shape, name = "input_b")
    
    if num_siam_layers > 0:
        siamese_layer = Dense(num_siam_nodes, activation = 'relu', kernel_regularizer = regularizers.l1(regularization), name = "siamese_0")

        hidden_a = siamese_layer(input_a)
        hidden_b = siamese_layer(input_b)

    for i in range(1, num_siam_layers):
        siamese_layer = Dense(num_siam_nodes, activation = 'relu', kernel_regularizer = regularizers.l1(regularization), name = "siamese_" + str(i))

        hidden_a = siamese_layer(hidden_a)
        hidden_b = siamese_layer(hidden_b)

    merged_layer = keras.layers.concatenate([hidden_a, hidden_b], axis = -1, name = "merged") # Why is the default axis -1?

    for i in range(num_hid_layers):
        merged_layer = Dense(num_hid_nodes, activation = 'relu', kernel_regularizer = regularizers.l1(regularization), name = "hidden_" + str(i))(merged_layer)

    output = Dense(output_nodes, activation = 'softmax', name = "output")(merged_layer)

    model = Model(inputs=[input_a, input_b], outputs=output)

    model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])
    
    return model



def simple_model(input_shape, output_nodes, num_hid_layers, num_hid_nodes, dropout = 0, regularization = 0):
    input_layer = Input(shape = input_shape, name = "input")
    
    if num_hid_layers > 0:
        hidden_layer = Dense(num_hid_nodes, activation = 'relu', kernel_regularizer = regularizers.l1(regularization), name = "hidden_0")(input_layer)
        hidden_layer = Dropout(dropout, name = "dropout_0")(hidden_layer)
    
    for i in range(1,num_hid_layers):
        hidden_layer = Dense(num_hid_nodes, activation = 'relu', kernel_regularizer = regularizers.l1(regularization), name = "hidden_" + str(i))(hidden_layer)
        hidden_layer = Dropout(dropout, name = "dropout_" + str(i))(hidden_layer)
        
    output = Dense(output_nodes, activation = 'softmax', name = "output")(hidden_layer)
    
    model = Model(inputs=input_layer, outputs = output)
    
    model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])
    
    return model