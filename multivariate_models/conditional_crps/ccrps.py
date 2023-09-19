
from univariate_models.crps.crps_loss_mixture_gaussian import mixture_gaussian_CRPS
import tensorflow as tf
import keras
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpm = tfp.math
from math import pi

# In case a variable has to be strictly positive

def activate_pos(arr):
    return 1e-2 + keras.activations.softplus(arr)

def activate_weights(arr):
    return 1e-2 + keras.activations.sigmoid(arr)

'''
    Own array normalization code, as keras.utils.normalize doesn't seem to normalize properly?
'''
def normalize(arr, axis):
    sum_arr = tf.reduce_sum(arr, axis=axis, keepdims=True)
    sum_arr = tf.repeat(sum_arr, arr.shape[axis], axis=axis)
    return arr / sum_arr

'''
    @input: k x l x n tensor
    @output two k x l x n x n tensors, with all possible combinations
'''
def comb_generator_2d_batch(arr):
    arr2 = tf.reshape(arr, (-1, arr.shape[1], arr.shape[2], 1))
    arr2 = tf.repeat(arr2, repeats = arr.shape[2], axis=3)
    arr2t = tf.transpose(arr2, perm=[0,1,3,2])
    return arr2, arr2t


'''
    @input: k x l x n x n tensor
    @output: k x l x n x (n-1) tensor with diagonal removed
'''
def remove_diagonal_2d_batch(arr):
    # Make the array smaller, else the lower selection won't work
    arr1 = tf.gather(arr, range(arr.shape[3]-1), axis=3)
    arr1 = tf.gather(arr1, range(1, arr.shape[2]), axis=2)
    # Now, keep the lower part
    arr1 = tf.linalg.LinearOperatorLowerTriangular(arr1).to_dense()
    
    # Make space for the upper part
    arr1 = tf.pad(arr1, tf.constant([[0,0],[0,0],[1,0],[0,0]]))
    
    # Make the matrix smaller, else the upper selection won't work
    arr2 = tf.gather(arr, range(1, arr.shape[3]), axis=3)
    arr2 = tf.gather(arr2, range(arr.shape[2]-1), axis=2)

    # Getting the upper matrix is achieved by transposing, getting the lower matrix, and transposing back
    arr2 = tf.linalg.LinearOperatorLowerTriangular(tf.transpose(arr2, perm=[0,1,3,2])).to_dense()
    arr2 = tf.transpose(arr2, perm=[0,1,3,2])
    
    # Making space for the lower part
    arr2 = tf.pad(arr2, tf.constant([[0,0],[0,0],[0,1],[0,0]]))
    return arr1 + arr2

'''
    Postprocess the model output
    @input: batch_size x m x (0.5 * n * (n+3)) tensor
    @output: means (batch_size x m x n) tensor
    @ouptut: cov (batch_size x m x n x n) tensor
    @output: weights (batch_size x m) tensor
'''
def postprocess_model_output(y_pred, n):
    weights = tf.gather(y_pred, 0, axis=2)      # First diminsion are the weights
    weights = activate_weights(weights)         # Activate them sigmoid. Not entirely neccesary, since we'll normalize them anyway.
    weights = normalize(weights, axis=1)        # Normalize the weights
    
    means = tf.gather(y_pred, range(1,n+1), axis=2)        # Mean vectors are the next n columns
    
    
    L_diag = tf.gather(y_pred, range(n+1, 2*n+1), axis=2)   # The diagonal entries of L are the non 
    #print("L_diag", L_diag)
    L_non_diag = tf.gather(y_pred, range(2*n+1, y_pred.shape[2]), axis=2)   #All other entries are the non-diagonal entries
    L_diag = activate_pos(L_diag)               # Diagonals should be strictly positive to guarantee positive-definite matrix
    
    L = tfpm.fill_triangular(L_non_diag)           #Set the non-diagonal entries.
    L = tf.pad(L, tf.constant([[0,0],[0,0],[1,0],[0,1]]))   # Increase the matrices from n-1 x n-1 to n x n
    L = tf.linalg.set_diag(L, L_diag)     #Now, fill in the diagonal
    cov = tf.matmul(L, L, transpose_b = True)       # The reverse Cholensky decomposition.
    return means, cov, weights

def merge_last_2_dims(arr):
    return tf.reshape(arr, shape=(-1, arr.shape[1], arr.shape[2] * arr.shape[3]))

'''
    Computes all conditional distributions
'''
def conditional_CRPS_mixtures(y1, y2, m1, m2, v1, v2, covs, weights):
    
    # Repeat the weights, so we can use them in tensor multiplications
    weights_rep = tf.reshape(weights, (-1, weights.shape[1], 1, 1))
    weights_rep = tf.repeat(weights_rep, m1.shape[2], axis=2)
    weights_rep = tf.repeat(weights_rep, m2.shape[3], axis=3)
    
    # Compute the conditional means and stdevs
    cond_means = m1 + covs/v2 * (y2 - m2)
    cond_stdevs = tf.math.sqrt(v1 - tf.pow(covs,2)/v2)
    
    # Compute the densities, making sure they're strictly positive to prevent zero-divisions during normalization
    dens_y2 = 1e-10 + tf.pow(2 * pi * v2,-0.5) * tf.math.exp(-0.5 * tf.pow(y2 - m2,2)/v2)
    cond_weights = normalize(dens_y2 * weights_rep, axis = 1)

    # Finally, fit them into the batch_size x m x dim format neccesary to apply the univariate mixture Gaussian CRPS code

    cond_stdevs = merge_last_2_dims(cond_stdevs)
    cond_weights = merge_last_2_dims(cond_weights)
    cond_means = merge_last_2_dims(cond_means)
    y1 = merge_last_2_dims(y1)
    y1 = tf.gather(y1, 0, axis=1)
    
    return y1, cond_means, cond_stdevs, cond_weights
    #print(cond_stdevs)
    

def make_biv_pairs(means, covs, weights, y_true):
    variances = tf.linalg.diag_part(covs)
    
    #Generate all possible combinations of 1 entry given another entry
    v1, v2 = comb_generator_2d_batch(variances)
    m1, m2 = comb_generator_2d_batch(means)
    
    y_true_rep = tf.reshape(y_true, (-1, 1, y_true.shape[1]))
    y_true_rep = tf.repeat(y_true_rep, m1.shape[1], axis=1)
    y1, y2 = comb_generator_2d_batch(y_true_rep)
    
    # We do not want to compare an index with itself, so we'll remove the diagonal from each matrix
    v1 = remove_diagonal_2d_batch(v1)
    v2 = remove_diagonal_2d_batch(v2)
    covs = remove_diagonal_2d_batch(covs)
    m1 = remove_diagonal_2d_batch(m1)
    m2 = remove_diagonal_2d_batch(m2)
    y1 = remove_diagonal_2d_batch(y1)
    y2 = remove_diagonal_2d_batch(y2)
    
    #print("m1",m1)
    #print("m2",m2)
    
    return conditional_CRPS_mixtures(y1, y2, m1, m2, v1, v2, covs, weights)

def compute_marginals(means, covs, weights, y_true):
    stdevs = tf.sqrt(tf.linalg.diag_part(covs))

    # Repeat the weights for all marginals
    weights = tf.reshape(weights, (-1, weights.shape[1], 1))
    weights = tf.repeat(weights, means.shape[2], axis=2)
    
    return y_true, means, stdevs, weights

def partial_mixture_CCRPS(y_true, y_pred):
    m = y_pred.shape[1]
    n = y_true.shape[1]

    means, covs, weights = postprocess_model_output(y_pred, n)

    # Compute all conditional distributions
    cond_y, cond_means, cond_stdevs, cond_weights = make_biv_pairs(means, covs, weights, y_true)
    
    # Compute all marginal distributions
    marg_y, marg_means, marg_stdevs, marg_weights = compute_marginals(means, covs, weights, y_true)
    
    y = tf.concat([cond_y, marg_y], axis=1)
    means = tf.concat([cond_means, marg_means], axis=2)
    stdevs = tf.concat([cond_stdevs, marg_stdevs], axis=2)
    weights = tf.concat([cond_weights, marg_weights], axis=2)
    return mixture_gaussian_CRPS(y, means, stdevs, weights)
