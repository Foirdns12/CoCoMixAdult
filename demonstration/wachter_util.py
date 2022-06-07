import numpy as np

import tensorflow as tf


def calculate_mad(data):
    """Calculate the Median Absolute Deviations for the dataset."""
    feature_median = np.nanmedian(data, axis=0)
    #print(feature_median)
    med=np.nanmedian(np.abs(data - feature_median), axis=0)
    #for i in len(med):
     #   if med[i]==0:
     #       med[i]=np.mean(np.abs(data[:,i]-feature_median[i]))
    return med


def median(v):
    v = tf.reshape(v, [-1])
    m = v.get_shape()[0] // 2
    return tf.reduce_min(tf.nn.top_k(v, m, sorted=False).values)


def convert_inputs(input_dict, input_types):
    return {k: tf.reshape(tf.convert_to_tensor([v], dtype=input_types[k]), (1, 1))
            for k, v in input_dict.items()}
