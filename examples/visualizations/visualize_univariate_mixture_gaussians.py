from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


"""
    Function used to convert Gaussian distributions to mixture Gaussian distributions of a single mixture.
    
    :param mus: an (n, dim) shaped numpy array, denoting the forecasted means
    :param sigmas: an (n, dim) shaped numpy array, denoting the forecasted standard deviations
    
    :return mus: an (n, 1, dim) shaped numpy array, denoting the forecasted means
    :return sigmas: an (n, 1, dim) shaped numpy array, denoting the forecasted standard devations
    :retun weights: an (n, 1, dim) shaped numpy array, filled with ones.
"""
def convert_to_mixture(mus, sigmas):
    mus = np.expand_dims(mus, axis=1)
    sigmas = np.expand_dims(sigmas, axis=1)
    weights = np.empty(mus.shape)
    weights.fill(1.)
    return mus, sigmas, weights



"""
    Makes simple visualizations of the forecasted Gaussian distributions and their observations
    
    This function has two possibilities for input, allowing for both Gaussian and mixture Gaussian distributions as input.
    
    :param y_true: an (n, dim) shaped numpy array, denoting the observations
    
    if weights is None:
        :param mus: an (n, dim) shaped numpy array, denoting the forecasted means
        :param sigmas: an (n, dim) shaped numpy array, denoting the forecasted standard deviations
    
    else:
        :param mus: an (n, n_mixtures, dim) shaped numpy array, denoting the forecasted means
        :param sigmas: an (n, n_mixtures, dim) shaped numpy array, denoting the forecasted standard devations
        :param weights: an (n, n_mixtures, dim) shaped numpy array, denoting the forecasted weights
"""

def visualize_gaussian_predictions(y_true, mus, sigmas, weights = None):
    n_targets = y_true.shape[1]
    
    # If a Gaussian distribution is inputted, make it a mixture Gaussian of 1 mixture
    if weights is None:
        mus, sigmas, weights = convert_to_mixture(mus, sigmas)
    
    # Swapping axes makes it easier to iterate over the mixtures
    mus = np.swapaxes(mus, 1, 2)
    sigmas = np.swapaxes(sigmas, 1, 2)
    weights = np.swapaxes(weights, 1, 2)
    
    # Iterate over each predicted distribution + observation
    for mu, sigma, weight, real in zip(mus, sigmas, weights, y_true):
        #Plot all dim target dimensions in different suplots
        fig, ax = plt.subplots(1, n_targets, figsize = (2.2 * n_targets, 2))
        fig.tight_layout()
        
        

        # We defined the model to output dim distributions. We'll print each in a subplot
        for idx, (m, s, w, r) in enumerate(zip(mu, sigma, weight, real)):
            
            # Take a lower bound of the lowest mixture and an upper bound of the highest mixture
            minVal = min(r, np.min(m - 3 * s))
            maxVal = max(r, np.max(m + 3 * s))
            
            x = np.linspace(minVal, maxVal, 500)
            
            # The pdf is defined as a weighted sum over the mixture's pdf's
            pred = sum([this_w * norm(this_m, this_s).pdf(x) for this_m, this_s, this_w in zip(m, s, w)])
            
            ax[idx].fill(x, pred, alpha = 0.3) # Plot the pdf
            ax[idx].plot(x, pred, color='black')
            ax[idx].plot([r, r], [0, max(pred)], '--', color='red') # Plot the observations
            ax[idx].set_xlabel("Value")
            ax[idx].set_ylabel("Density")
            ax[idx].grid()
    plt.show()
    
    
    
"""
    Compute a Probability Integral Transformation of the forecasted Gaussian distributions and observations
    and makes a plot showing the PIT'd quantiles for each dimension.
    
    This function has two possibilities for input, allowing for both Gaussian and mixture Gaussian distributions as input.
    
    :param y_true: an (n, dim) shaped numpy array, denoting the observations
    
    if weights is None:
        :param mus: an (n, dim) shaped numpy array, denoting the forecasted means
        :param sigmas: an (n, dim) shaped numpy array, denoting the forecasted standard deviations
    
    else:
        :param mus: an (n, n_mixtures, dim) shaped numpy array, denoting the forecasted means
        :param sigmas: an (n, n_mixtures, dim) shaped numpy array, denoting the forecasted standard devations
        :param weights: an (n, n_mixtures, dim) shaped numpy array, denoting the forecasted weights
"""
def do_gaussian_PIT(y_true, mus, sigmas, weights = None):
    
    n_targets = y_true.shape[1]
    
    # If a Gaussian distribution is inputted, make it a mixture Gaussian of 1 mixture
    if weights is None:
        mus, sigmas, weights = convert_to_mixture(mus, sigmas)
        
    # Swapping axes makes it easier to iterate over the plots
    mus = np.swapaxes(mus, 1, 2)
    sigmas = np.swapaxes(sigmas, 1, 2)
    weights = np.swapaxes(weights, 1, 2)
    
    # Data structure that will contain all PIT points
    qs = [[] for i in range(n_targets)]
    
    # Go through each pair
    for mu, sigma, weight, real in zip(mus, sigmas, weights, y_true): #Iterate over each (x, y) pair
        
        # Iterate over each of the dim distributions forecasted
        for idx, (m, s, w, r) in enumerate(zip(mu, sigma, weight, real)):
            
            # The cdf is defined as a weighted sum over the mixture's cdf's
            cdf = sum([this_w * norm(this_m, this_s).cdf(r) for this_m, this_s, this_w in zip(m, s, w)])
            qs[idx].append(cdf)
            
    plt.figure()
    for idx, q in enumerate(qs):
        plt.plot(np.linspace(0,1,len(q)), sorted(q), label=f"Dim {idx+1}")
        plt.xlabel("Uniform quantiles")
        plt.ylabel("PIT quantiles")
        plt.legend()
        plt.grid()
    plt.show()