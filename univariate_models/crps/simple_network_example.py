import keras
from keras.layers import Dense
import numpy as np
from numpy.random import normal

from crps_loss_gaussian import gaussian_CRPS_loss
from visualizations import visualize_gaussian_predictions


"""
     We define a simple example model, with two hidden layers and "adam" loss.
"""
def gaussian_crps_model(import_dim, target_dim):
    model = keras.Sequential()
    model.add(keras.Input(shape=(import_dim)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(2 * target_dim))
    model.summary()
    model.compile(loss = gaussian_CRPS_loss, optimizer="adam")
    return model



"""
    Generates a dummy dataset. Entries in x and y are related (sampled from identical distributions)
    but y's entries warped by linear mappings
"""
def generate_dummy_data(input_dim, target_dim, n):
    x = np.empty((n, input_dim))
    y = np.empty((n, target_dim))
    for idx, (x_val, y_val) in enumerate(zip(x, y)):
        random_mean = normal(loc = 0, scale = 1)
        random_stdev = np.abs(normal(loc = 1, scale = 1)) + 1e-5 # Small addition to ensure strict positivity
        samples = normal(loc = random_mean, scale = random_stdev, size = input_dim + target_dim)
        this_x = samples[:input_dim]
        this_y = [(i+1) * s - i for i, s in enumerate(samples[input_dim:])]
        x[idx,:] = this_x
        y[idx,:] = this_y
    return x, y


"""
    In this example, we define a simple ANN model, and train it via Gaussian CRPS loss.
    We will then visualize the predictions
"""
if __name__ == "__main__":
    input_dim = 10
    target_dim = 5
    
    epochs = 10
    
    n_train = 10000
    n_test = 1000
    
    model = gaussian_crps_model(input_dim, target_dim)
    x_train, y_train = generate_dummy_data(input_dim, target_dim, n_train)
    x_test, y_test = generate_dummy_data(input_dim, target_dim, n_test)

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs)
    
    examples = 5
    y_pred = model.predict(x_test[:5])
    visualize_gaussian_predictions(y_test[:5], y_pred)