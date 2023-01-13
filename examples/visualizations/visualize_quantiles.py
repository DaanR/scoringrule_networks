import matplotlib.pyplot as plt
import numpy as np

"""
Simple quantile regression CDF computation, via the 'step-wise' definition

TODO: add documentation
"""
def get_cdf(quantiles, y):
    quantiles = list(quantiles)
    # Return the ratio of points that are smaller
    return len([q for q in quantiles if q <= y])/len(quantiles)

def visualize_quantiles(y_true, all_quantiles):
    n_targets = all_quantiles.shape[1]
    
    for y, quantiles in zip(y_true, all_quantiles):
        fig, ax = plt.subplots(1, n_targets, figsize = (2.2 * n_targets, 2))
        fig.tight_layout()
        
        # Iterate over each of the targets and their predicted quantiles:
        for idx, (this_y, this_quantiles) in enumerate(zip(y, quantiles)):
            minVal = min(np.min(this_quantiles), this_y) - 0.2
            maxVal = max(np.max(this_quantiles), this_y) + 0.2
            
            x = np.linspace(minVal, maxVal, 500)
            pred = [get_cdf(this_quantiles, x_val) for x_val in x]
            zeros = np.zeros(x.shape)
            
            ax[idx].fill_between(x, zeros, pred, alpha = 0.3) # Plot the cdf
            ax[idx].plot(x, pred, color='black')
            ax[idx].plot([this_y, this_y], [0, 1], '--', color='red') # Plot the observations
            ax[idx].set_xlabel("Value")
            ax[idx].set_ylabel("Cumulative density")
            ax[idx].grid()
    plt.show()