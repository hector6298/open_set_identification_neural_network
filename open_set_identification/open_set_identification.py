import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import model 
import loss_function
import matplotlib.pyplot as plt
import visualize
import outlier_score
from steps_definitions import *

#MNIST dataset loading
mnist = tf.keras.datasets.mnist
print("starting data loading")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# Add a channels dimension

x_train = tf.cast(x_train[..., tf.newaxis],tf.float32)[:2000]
x_test = tf.cast(x_test[..., tf.newaxis],tf.float32)[:500]
y_train = y_train[:2000]
y_test = y_test[:500]

indices_train = tf.where(tf.less_equal(y_train,5))
indices_test = tf.where(tf.less_equal(y_test,5))
x_train = tf.squeeze(tf.gather(x_train,indices_train),1)
y_train = tf.squeeze(tf.gather(y_train,indices_train))

x_test = tf.squeeze(tf.gather(x_test,indices_test),1)
y_test = tf.squeeze(tf.gather(y_test,indices_test))


#shuffling and batching dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(100)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(250)
print('finished loading')

def accuracy(true,preds):
    correct = tf.equal(true,preds)
    test_accuracy(correct)
    

def train_step_representation(batch,labels_batch):
    with tf.GradientTape() as tape:
        predictions = Model(batch, training = True)
        iiloss = tf.convert_to_tensor( loss_function.ii_loss(predictions,labels_batch),dtype= tf.float32)
        #print(model.trainable_variables)
    gradients = tape.gradient(iiloss, Model.trainable_variables)
    optimizer_rep.apply_gradients(zip(gradients,Model.trainable_variables))
    train_loss_rep(iiloss)

def train_step_classification(batch, labels_batch):
    with tf.GradientTape() as tape:
        
        predictions = classification_model(batch, training = True)
        loss = loss_object(labels_batch, predictions)
    gradients = tape.gradient(loss, classification_model.trainable_variables)
    optimizer_class.apply_gradients(zip(gradients, classification_model.trainable_variables))
    train_loss_class(loss)
    train_accuracy_class(labels_batch, predictions)

#@tf.function
#Cannot use tf.function as batch is being pythonically iterated in loss function
def test_step(images, labels, class_means):
    global threshold
    predictions = Model(images, training = False)
    embed = predictions.numpy()
    visualize.visualize(embed,labels.numpy())
    t_loss = loss_function.ii_loss(predictions,labels)
    scores_test = outlier_score.compute_outlier_score(predictions,class_means)
    print(scores_test)
    results_known_unknown = np.where(scores_test > threshold)
    class_preds = classification_model(images, training = False)
    class_preds = tf.argmax(class_preds,axis = 1).numpy()
    class_preds[results_known_unknown] = 7
    class_preds = tf.convert_to_tensor(class_preds)
    class_preds = tf.cast(class_preds, tf.uint8)
    labels = tf.cast(labels,tf.uint8)
    print(class_preds)
    accuracy(labels, class_preds)


def threshold_step(images, class_means):
    global scores
    predictions = Model(images)
    scores_batch = outlier_score.compute_outlier_score(predictions,class_means)
    scores = np.append(scores, scores_batch)



Model = model.Open_ANN()
classification_model = model.classificationNN()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer_rep = tf.keras.optimizers.SGD()
optimizer_class = tf.keras.optimizers.Adam()

train_loss_rep = tf.keras.metrics.Mean(name='train_loss_rep')

train_loss_class = tf.keras.metrics.Mean(name='train_loss_class')
train_accuracy_class = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy_class')


test_accuracy = tf.keras.metrics.Mean(name='test_accuracy_class')


#@tf.function

print('starting training')
EPOCHS = 20

for epoch in range(EPOCHS):
  for images, labels in train_ds:
    train_step_representation(images, labels)
    train_step_classification(images,labels)

  template = 'Epoch {}, ii-Loss: {}   ||  ce-loss: {}, trainning-classification accuracy: {}'
  print (template.format(epoch+1,
                         train_loss_rep.result(),
                         train_loss_class.result(),
                         train_accuracy_class.result()))

means_dict, class_means = loss_function.compute_global_means(train_ds, Model)
for images, labels in train_ds:
    preds = Model(images, training = False)
    visualize.visualize(preds.numpy(),labels.numpy())

scores = np.array([])
for images, labels in train_ds:
    threshold_step(images,class_means)
scores = np.sort(np.squeeze(np.reshape(scores,[1,-1])))
print(scores)
threshold = np.percentile(scores, 99)
print(threshold)

print('starting test')
for test_images, test_labels in test_ds:
    test_step(test_images, test_labels,class_means)

    template = 'test accuracy: {}'
    print(template.format(test_accuracy.result()))

#after training, compute class means of the entire training dataset
# after that define another step (threshold_step) in wich we iterate
#through the entire training ds and compute a threshold
#run test step with these threshold

#===================================================================

  