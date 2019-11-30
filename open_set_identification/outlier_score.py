import numpy as np
import tensorflow as tf

def compute_outlier_score(test_batch, means_vector):
    diff = np.zeros(test_batch.shape[0])+ 1e6
    for i in range(len(test_batch)):
        for j in range(len(means_vector)):
            t = tf.square(tf.norm(tf.subtract(means_vector[j] ,test_batch[i]))).numpy()
            if t < diff[i]:
                diff[i] = t
    return diff

