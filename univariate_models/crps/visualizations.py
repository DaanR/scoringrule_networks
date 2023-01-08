from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

from crps_loss_gaussian import get_mu_sigma


"""
    Makes simple visualizations of the forecasted Gaussian distributions and their observations
"""

def visualize_gaussian_predictions(y_true, y_pred):
    dim = y_true.shape[1]
    
    #The get_mu_sigma function converts the predictions to the means and standard devations
    mus, sigmas = get_mu_sigma(y_pred, dim)
    
    for mu, sigma, real in zip(mus, sigmas, y_true): #Iterate over each (x, y) pair
        #Plot all dim target dimensions in different suplots
        fig, ax = plt.subplots(1, dim, figsize = (2.2 * dim, 2))
        fig.tight_layout()
        for idx, (m, s, r) in enumerate(zip(mu, sigma, real)):
            minVal = min(r, m - 3 * s)
            maxVal = max(r, m + 3 * s)
            x = np.linspace(minVal, maxVal, 500)
            pred = norm(m, s).pdf(x)
            
            ax[idx].fill(x, pred, alpha = 0.3) # Plot the pdf
            ax[idx].plot(x, pred, color='black')
            ax[idx].plot([r, r], [0, max(pred)], '--', color='red') # Plot the observations
            ax[idx].set_xlabel("Value")
            ax[idx].set_ylabel("Density")
            ax[idx].grid()
    plt.show()