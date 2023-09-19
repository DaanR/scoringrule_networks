def energy_score_loss(y_true, y_pred):
    eps = 1e-30 # Neccesary as else gradients become undefiend when dist = 0 (gradient of sqrt(0) is undefined)
    d = y_true.shape[1]
    n_points = y_pred.shape[1]
        
    y_true = tf.reshape(y_true, (-1, 1, y_true.shape[1]))
    y_true = tf.repeat(y_true, n_points, axis=1)
    
    dists1 = tf.sqrt(tf.reduce_sum(tf.pow(y_true - y_pred,2) + eps, axis=2))

    y_pred2 = tf.reshape(y_pred, (-1, y_pred.shape[1], 1, y_pred.shape[2]))
    y_pred2 = tf.repeat(y_pred2, n_points, axis=2)
    y_pred3 = tf.transpose(y_pred2, perm=[0, 2, 1, 3])
    dists2 = tf.sqrt(tf.reduce_sum(tf.pow(y_pred2 - y_pred3,2) + eps, axis=3))
    
    # Compute all energy scores at the same time
    return tf.reduce_mean(dists1) - 0.5 * tf.reduce_mean(dists2)
