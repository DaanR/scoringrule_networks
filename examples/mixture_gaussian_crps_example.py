import keras
from keras.layers import Dense, Reshape

import sys
sys.path.append('..') # Add the parent folder

from univariate_models.crps.crps_loss_mixture_gaussian import mixture_gaussian_CRPS_loss 
from univariate_models.mixture_gaussian_helpers import preprocess_mixture_output
from visualizations.visualize_univariate_mixture_gaussians import visualize_gaussian_predictions, do_gaussian_PIT


from example_data_generation.generate_example_data import generate_bim_gaussian_univar_data

"""
     Defines a simple example model, with three hidden layers and "adam" loss.
     
     :param input_dim: an integer, denoting the size of the input layer
     :param output_dim: an integer, denoting the dimensionality of the target distribution
     :n_mixture: an integer, denoting the specified number of mixtures.
     
     :return: a keras model
"""
def gaussian_crps_model(import_dim, target_dim, n_mixtures):
    model = keras.Sequential()
    model.add(keras.Input(shape=(import_dim)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(3 * n_mixtures * target_dim))
    model.add(Reshape((n_mixtures, 3 * target_dim)))
    model.summary()
    model.compile(loss = mixture_gaussian_CRPS_loss, optimizer="adam")
    return model



"""
    In this example, we define a simple ANN model, and train it via Gaussian CRPS loss.
    We will then visualize the predictions, and compute probability integral transformations
"""
if __name__ == "__main__":
    input_dim = 10
    target_dim = 5
    n_mixtures = 2
    
    epochs = 15
    
    n_train = 10000
    n_test = 1000
    
    ''' Defining the model '''
    model = gaussian_crps_model(input_dim, target_dim, n_mixtures)
    
    ''' Generating dummy data '''
    x_train, y_train = generate_bim_gaussian_univar_data(input_dim, target_dim, n_train)
    x_test, y_test = generate_bim_gaussian_univar_data(input_dim, target_dim, n_test)
    
    print("Data shapes")
    print("x_train", x_train.shape, "y_train", y_train.shape)
    print("x_test", x_test.shape, "y_test", y_test.shape)

    ''' Training the model '''
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs)
    
    ''' Making predictions '''
    y_pred = model.predict(x_test)
      
    # Convert the predictions to tensors of denoting the means, standard deviations and weights
    mus, sigmas, weights = preprocess_mixture_output(y_pred, target_dim)
    
    # Convert the tensors to numpy arrays, which is easier for visualization
    mus = mus.numpy()
    sigmas = sigmas.numpy()
    weights = weights.numpy()
    
    ''' Visualizing the output '''
    examples = 5
    visualize_gaussian_predictions(y_test[:examples], mus[:examples], sigmas[:examples], weights[:examples])
#    
    ''' Probability integral transformations'''
    do_gaussian_PIT(y_test, mus, sigmas, weights)