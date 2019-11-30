import numpy as np
import tensorflow as tf


def compute_global_means(dataset, Model):
    mean_computer = tf.keras.metrics.MeanTensor()
    means_dict = dict()
    means = []
    for images, labels in dataset:
        preds = Model(images, training = False)
        for i in range(len(preds)):
            if labels[i].numpy() not in means_dict:
                means_dict[labels[i].numpy()] = [preds[i]]
            else:
                means_dict[labels[i].numpy()].append(preds[i])
    for key in means_dict:
        for tensor in means_dict[key]:
            mean_computer.update_state(tensor)
        class_mean = mean_computer.result()
        means_dict[key] = class_mean
        means.append(class_mean)
        mean_computer.reset_states()
    return means_dict, means

def compute_class_means(batch,labels_batch):
    mean_computer = tf.keras.metrics.MeanTensor()
    means_dict = dict()
    means = []
    for i in range(len(batch)):
        if labels_batch[i].numpy() not in means_dict:
            means_dict[labels_batch[i].numpy()] = [batch[i]]
        else:
            means_dict[labels_batch[i].numpy()].append(batch[i])
    for key in means_dict:
        for tensor in means_dict[key]:
            mean_computer.update_state(tensor)
        class_mean = mean_computer.result()
        means_dict[key] = class_mean
        means.append(class_mean)
        mean_computer.reset_states()

    return means_dict, means
    
def compute_intra_spread(batch,labels_batch, means_dict):
    intra_spread = 0
   # print(tf.square(tf.norm((means_dict[labels_batch[0].numpy()] - batch[0]))))
    for i in range(len(batch)):
        t = tf.square(tf.norm((means_dict[labels_batch[i].numpy()] - batch[i])))
        intra_spread += t
    return intra_spread/len(batch)

def compute_inter_separation(means_vector):
    diff = 1e6
    for i in range(len(means_vector)-1):
        for j in range(i+1,len(means_vector)):
            t = tf.square(tf.norm((means_vector[i] - means_vector[j])))
            if t < diff:
                diff = t
    return diff

def ii_loss(batch,labels_batch):
    means_dict, means_vector = compute_class_means(batch,labels_batch)
    return compute_intra_spread(batch,labels_batch,means_dict) - compute_inter_separation(means_vector)
