import tensorflow as tf
import numpy as np

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
        loss = loss_object(labels_batch, batch)
    gradients = tape.gradient(loss, classification_model.trainable_variables)
    optimizer_class.apply_gradients(zip(gradients, classification_model.trainable_variables))
    train_loss_class(loss)
    train_accuracy_class(labels_batch, predictions)

#@tf.function
#Cannot use tf.function as batch is being pythonically iterated in loss function
def test_step(images, labels, class_means):
    global threshold
    predictions = model(images, training = False)
    embed = predictions.numpy()
    visualize.visualize(embed,labels.numpy())
    t_loss = loss_function.ii_loss(predictions,labels)
    scores_test = outlier_score.compute_outlier_score(predictions,class_means)
    results_known_unknown = np.where(scores_test > threshold)
    class_preds = classification_model(images, training = False)
    class_preds = tf.argmax(class_preds,axis = 1).numpy()
    class_preds[results_known_unknown] = 7
    class_preds = tf.convert_to_tensor(class_preds)
    accuracy(labels, class_preds)

    


    #check with synthetic example only using classification model
    

def threshold_step(images, class_means):
    global scores
    predictions = Model(images)
    scores_batch = outlier_score.compute_outlier_score(predictions,class_means)
    scores = np.append(scores, scores_batch)

