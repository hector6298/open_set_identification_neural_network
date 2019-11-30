
import tensorflow as tf
import numpy as np
import matplotlib

import matplotlib.pyplot as plt

def visualize(embed, labels):

    labelset = set(labels.tolist())

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

    #fig, ax = plt.subplots()
    for label in labelset:
        indices = np.where(labels == label)
        ax.scatter(embed[indices,0], embed[indices,2], label = label, s = 20)
    ax.legend()
    plt.show()
    

