import matplotlib.pyplot as plt
import numpy as np

"""
    Simple quantile induced CDF computation, via the 'step-wise' definition

    :input quantiles: a sorted (n_quantiles,) shaped numpy array of quantiles
    :input y: the value for which we want to evaluted the quantile induced CDF
    
    :return the cdf value (in [0,1])
"""
def get_cdf(quantiles, y):
    quantiles = list(quantiles)
    # Return the ratio of points that are smaller
    return len([q for q in quantiles if q <= y])/len(quantiles)

"""
    Simple quantile induced CDF computation, via the 'step-wise' definition

    :input quantiles: a sorted (n_quantiles,) shaped numpy array of quantiles
    :input y: the value for which we want to evaluted the quantile induced CDF
    :input minVal: a lower bound for the quantiles. If None, a sensible guess is made
    :input maxVal: a lower bound for the quantiles. If None, a sensible guess is made
    
    :return the cdf value (in [0,1])
"""
def get_cdf_smoothed(quantiles, y, minVal = None, maxVal = None):
    quantiles = list(quantiles)
    
    # Add sensible minimum and maximum values
    if minVal is None or maxVal is None:
            interval = max(quantiles) - min(quantiles)
            minVal = min(quantiles) -  1./len(quantiles) * interval
            maxVal = max(quantiles) +  1./len(quantiles) * interval
    
    quantiles = [minVal] + quantiles + [maxVal]
    
    minIdx = len([q for q in quantiles if q < y])
    
    # If y is smaller or bigger than the smallest/biggest value, don't interpolate, but return 0 or 1
    if minIdx == len(quantiles):
        return 1.
    if minIdx == 0:
        return 0.
    
    # If two forecasted quantiles happen to be identical
    if quantiles[minIdx-1] == quantiles[minIdx]:
        return quantiles[minIdx]
    
    # If all edge cases are averted, do linear interpolation
    y1 = quantiles[minIdx-1]
    y2 = quantiles[minIdx]
    cdf1 = (minIdx - 1)/(len(quantiles)-1)
    cdf2 = minIdx/(len(quantiles)-1)
    
    this_cdf = cdf1 + (y - y1) * (cdf2 - cdf1)/(y2 - y1)
    if this_cdf < 0 or this_cdf > 1:
        print("------")
        print(this_cdf)
        print(quantiles)
        print(y1, y2, cdf1, cdf2)
        print(y)
    return this_cdf
    
    

"""
    Visualizes a quantile forecast, by generating n_forecasts * n_target plots
    
    :input y_true: an (n_forecasts, n_targets) shaped numpy array of observations
    :input all_quantiles: an (n_forecasts, n_targets, n_quantiles) numpy array tensor of quantile forecasts (last dimension is sorted)
    :input smoothed: boolean, if True, the get_cdf_smoothed function is used, else the get_cdf function is used
"""
def visualize_quantiles(y_true, all_quantiles, smoothed = False):
    n_targets = all_quantiles.shape[1]
    
    for y, quantiles in zip(y_true, all_quantiles):
        fig, ax = plt.subplots(1, n_targets, figsize = (2.2 * n_targets, 2))
        fig.tight_layout()
        
        # Iterate over each of the targets and their predicted quantiles:
        for idx, (this_y, this_quantiles) in enumerate(zip(y, quantiles)):
            this_quantiles = sorted(this_quantiles) #Precaution, in theory, the quantiles should be sorted
            # Determing plot boundaries
            minVal = min(np.min(this_quantiles), this_y)
            maxVal = max(np.max(this_quantiles), this_y)
            
            interval = maxVal - minVal
            minVal -=  2./len(this_quantiles) * interval
            maxVal +=  2./len(this_quantiles) * interval
            
            x = np.linspace(minVal, maxVal, 500)
            if smoothed:
                pred = [get_cdf_smoothed(this_quantiles, x_val) for x_val in x]
            else:
                pred = [get_cdf(this_quantiles, x_val) for x_val in x]
            zeros = np.zeros(x.shape)
            
            ax[idx].fill_between(x, zeros, pred, alpha = 0.3) # Plot the cdf
            ax[idx].plot(x, pred, color='black')
            #ax[idx].plot([this_y, this_y], [0, 1], '--', color='red') # Plot the observations
            ax[idx].set_xlabel("Value")
            ax[idx].set_ylabel("Cumulative density")
            ax[idx].grid()
    plt.show()


"""
    Visualizes the probability integral transformation for quantile distributions.
    
    :input y_true: an (n_forecasts, n_targets) shaped numpy array of observations
    :input all_quantiles: an (n_forecasts, n_targets, n_quantiles) numpy array tensor of quantile forecasts (last dimension is sorted)
    :input smoothed: boolean, if True, the get_cdf_smoothed function is used, else the get_cdf function is used
"""  
def quantile_PIT(y_true, all_quantiles, smoothed = False):
    n_targets = all_quantiles.shape[1]
    
    # Data structure that will contain all PIT points
    qs = [[] for i in range(n_targets)]
    
    for y, quantiles in zip(y_true, all_quantiles):
        
        # Iterate over each of the dim distributions forecasted
        for idx, (this_y, this_quantiles) in enumerate(zip(y, quantiles)):
            this_quantiles = sorted(this_quantiles) # Precaution, in theory, the quantiles should be sorted
            if smoothed:
                cdf = get_cdf_smoothed(this_quantiles, this_y)
            else:
                cdf = get_cdf(this_quantiles, this_y)
            qs[idx].append(cdf)
            
    plt.figure()
    for idx, q in enumerate(qs):
        plt.plot(np.linspace(0,1,len(q)), sorted(q), label=f"Dim {idx+1}")
        plt.xlabel("Uniform quantiles")
        plt.ylabel("PIT quantiles")
        plt.legend()
        plt.grid()
    plt.show()