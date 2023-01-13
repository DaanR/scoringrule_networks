import tensorflow as tf

"""
    Implementation of the quantile loss function. As instructed by https://onlinelibrary.wiley.com/doi/abs/10.1002/1099-131X%28200007%2919%3A4%3C299%3A%3AAID-FOR775%3E3.0.CO%3B2-V
    This function returns twice the quantile loss, such that it is comparable to CRPS. https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.1891
    
    :param y_true: a (batch_size, n_targets) shaped tensor
    :param y_pred: a (batch_size, n_targets, n_quantiles) shaped tensor
    
    :return: twice the quantile loss.
"""
def double_quantile_loss(y_true, y_pred):
    
    # Note, this list approach is known to be less efficient. 
    q = tf.constant([(2*i + 1)/(2*y_pred.shape[2]) for i in range(y_pred.shape[2])])
    
    y_true = tf.transpose(tf.stack([y_true]),perm=[1,2,0])
    u = y_true - y_pred

    pos = u * q
    neg = u * (q - 1)
    
    # One of them will be < 0, so replacing the positivity operator can be done like this.
    # It saves some "nasty" typecasting tricks.
    maxs = tf.math.maximum(pos, neg)
    
    # Multiply the final result by 2 to equalize it to CRPS
    return 2*tf.reduce_mean(maxs)

