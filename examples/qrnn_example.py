import keras
from keras.layers import Dense, Reshape

import sys
sys.path.append('..') # Add the parent folder

from univariate_models.quantile_regression.quantile_loss import double_quantile_loss

from example_data_generation.generate_example_data import generate_gaussian_univar_data, generate_bim_gaussian_univar_data


"""
     Defines a simple example model, with two hidden layers and "adam" loss.
"""
def qrnn_model(import_dim, target_dim, n_quantiles):
    model = keras.Sequential()
    model.add(keras.Input(shape=(import_dim)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(target_dim * n_quantiles))
    model.add(Reshape((target_dim, n_quantiles)))
    model.summary()
    model.compile(loss = double_quantile_loss, optimizer="adam")
    return model




def do_experiment(model, x_train, y_train, x_test, y_test):
    ''' Training the model '''
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs)
    
    ''' Making predictions '''
    y_pred = model.predict(x_test)
    
    # Convert the predictions to tensors of denoting the means and standard deviations
    #mus, sigmas = get_mu_sigma(y_pred, target_dim) # mus and sigmas are identically shaped to y_test
    
    # Convert them to numpy arrays. Makes visualization easier
    #mus = mus.numpy()
    #sigmas = sigmas.numpy()
    
    ''' Visualizing the output '''
    examples = 5
    #visualize_gaussian_predictions(y_test[:examples], mus[:examples], sigmas[:examples])
    
    ''' Probability integral transformations'''
    #do_gaussian_PIT(y_test, mus, sigmas)
    
    
    
    
    

if __name__ == "__main__":
    input_dim = 10
    target_dim = 5
    nr_quantiles = 20
    
    epochs = 15
    
    n_train = 10000
    n_test = 1000
    
    
    '''
        This first experiment is aimed at showing how Gaussian CRPS works with perfect data.
        We enter Gaussian distributed data into the model, and we find good distributions.
    '''
    print("Experiment 1: Gaussian target distributions, with Gaussian model")
    
    
    ''' Defining the model '''
    model = qrnn_model(input_dim, target_dim, nr_quantiles)
    
    ''' Generating dummy data '''
    x_train, y_train = generate_gaussian_univar_data(input_dim, target_dim, n_train)
    x_test, y_test = generate_gaussian_univar_data(input_dim, target_dim, n_test)
    
    print("Data shapes")
    print("x_train", x_train.shape, "y_train", y_train.shape)
    print("x_test", x_test.shape, "y_test", y_test.shape)
    
    do_experiment(model, x_train, y_train, x_test, y_test)
    
    
    
    '''
        Our second experiment is aimed at showing how PIT's detect a non-Gaussian target distribution
        We enter bimodal Gaussian distributed data into the model, but we still use our Gaussian model.
    '''
    print("Experiment 2: bimodal Gaussian target distributions, with Gaussian model")
    
    ''' Defining the model '''
    model = qrnn_model(input_dim, target_dim, nr_quantiles)
    
    ''' Generating dummy data '''
    x_train, y_train = generate_bim_gaussian_univar_data(input_dim, target_dim, n_train)
    x_test, y_test = generate_bim_gaussian_univar_data(input_dim, target_dim, n_test)
    
    print("Data shapes")
    print("x_train", x_train.shape, "y_train", y_train.shape)
    print("x_test", x_test.shape, "y_test", y_test.shape)
    
    # We should now see a significantly worse PIT (= not a 0-1 straight line).
    do_experiment(model, x_train, y_train, x_test, y_test)