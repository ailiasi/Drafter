import models

def fit_and_save_history(save_file, model, inputs, outputs, validation_data, epochs):
    # TODO: save
    history = model.fit(inputs, outputs, validation_data = validation_data, epochs = epochs)
    