
# coding: utf-8

# ## Network Relgularization(CNN)(EXERCISE 2)
# 
# In this code, I have implemented Q2 in which we had to extend the architecture of given in Q1. Specifically, we used dropout Techniques to the training.i added batch normalization layer and train themodel with tf.train.GradientDescentOptimizer.
# 
# #### First we load the relevant libraries

# In[37]:


from urllib.request import urlretrieve
from os.path import isfile, isdir
import tarfile
import pickle
import numpy as np
import matplotlib.pyplot as plt

cifar10_dataset_folder_path = 'cifar-10-batches-py'


# #### The following function loads the data from the batch file and reshapes the data.

# In[38]:


def load_cfar10_batch(batch_id):
    with open('data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
        
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
        
    return features, labels


# #### Normalize function to normalize the values between 0 and 1
# In[39]:


def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x 
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x


# #### We perform one-hot encoding to the labels.

# In[40]:


def one_hot_encode(x):
    """
        argument
            - x: a list of labels
        return
            - one hot encoding matrix (number of labels, number of class)
    """
    encoded = np.zeros((len(x), 10))
    
    for idx, val in enumerate(x):
        encoded[idx][val] = 1
    
    return encoded


# ## Preprocess all the data
# 
# #### The code cell below uses the previously implemented functions, normalize and one_hot_encode, to preprocess the given dataset.

# In[41]:


def preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode):
    n_batches = 5
    valid_features = []
    valid_labels = []
    
    c10_train_dataset, c10_train_labels = [], []

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(batch_i)
        
        features = normalize(np.array(features))
        labels = one_hot_encode(np.array(labels))
    
        c10_train_dataset.append(features)
        c10_train_labels.append(labels)
            
        # find index to be the point as validation data in the whole dataset of the batch (10%)
        index_of_validation = int(len(features) * 0.1)
        
        '''
        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])
        '''

    # load the test dataset
    with open('test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # preprocess the testing data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']
    
    test_features = normalize(np.array(test_features))
    test_labels = one_hot_encode(np.array(test_labels))
    
    return c10_train_dataset, c10_train_labels, test_features, test_labels


# #### Call the function and load the data

# In[42]:


c10_train_dataset, c10_train_labels, c10_test_dataset,c10_test_labels = preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)


# In[43]:


print(len(c10_test_dataset))


# #### In the following cells we will setup our structure of CNN And Data Augmentation 

# In[44]:

IMAGE_SIZE=32

def flip_images(X_imgs):
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X: img})
            X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype = np.float32)
    return X_flip
	


import tensorflow as tf
import math
# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

degrees=90
# Inputs
x = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3), name='input_x')
y =  tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# #### <<<  I used the ReLu activation function >>> Dropout Technique

# In[ ]:


import tensorflow as tf

def conv_net(x, keep_prob):
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))
    bias1 = tf.Variable(tf.constant(0.05, shape=[64]))

    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')
    conv1 += bias1
    
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv1 = tf.nn.relu(conv1_pool)
    #  normalization
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    conv2_pool = tf.nn.max_pool(conv1_bn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    flat = tf.contrib.layers.flatten(conv1_bn)  
# dropout 

    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=64, activation_fn=tf.nn.relu)
    #full1 = tf.nn.dropout(full1, keep_prob)
    
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
    #full2 = tf.nn.dropout(full2, keep_prob)
    
    out = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=10, activation_fn=tf.nn.softmax)
    return out


# ### Hyperparameters
# 
# * `epochs`: number of iterations until the network stops learning or start overfitting
# * `batch_size`: highest number that your machine has memory for.  Most people set them to common sizes of memory:
# * `keep_probability`: probability of keeping a node using dropout
# * `learning_rate`: number how fast the model learns

# In[ ]:


LOGDIR = "C:\\Users\\stech\\Downloads\\lab 6 (1)\\lab 6\\cifar-10-batches-py\\tensorboard"
LOGDIR="C:\\Users\\stech\\Documents\\tensorboard"
epochs = 10
batch_size = 128
#keep_probability = 0.9
learning_rate = 0.001
        


#  #### We initialize our CNN, define the GradientDescentoptimizer and loss function and also accuracy

# In[ ]:


logits = conv_net(x, keep_prob)
model = tf.identity(logits, name='logits') # Name logits Tensor, so that can be loaded from disk after training


# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy_train = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy_train')
accuracy_test = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy_test')

tf.summary.histogram('accuracy_train',accuracy_train)
tf.summary.histogram('accuracy_test',accuracy_test)
tf.summary.histogram("loss", cost)

tf.summary.scalar("loss", cost)
tf.summary.scalar("accuracy_train", accuracy_train)
tf.summary.scalar("accuracy_test", accuracy_test)


# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()  


# #### This function performs the learning/optimization using the training data

# In[ ]:


def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    session.run(optimizer, 
                feed_dict={
                    x: feature_batch,
                    y: label_batch,
                    keep_prob: keep_probability
                })


# ### Show Stats
# 
# #### The function print_stats runs the cost function. Accuracy function is also run on training and testing data

# In[ ]:


global count
count = 0

def print_stats(session, feature_batch, label_batch, cost, batch_features_test, 
                batch_labels_test, merged_op, sum_writer):
    global count
    loss = sess.run(cost, 
                    feed_dict={
                        x: feature_batch,
                        y: label_batch,
                        keep_prob: 1.
                    })
    valid_acc = sess.run(accuracy_train, 
                         feed_dict={
                             x: feature_batch,
                             y: label_batch,
                             #x: valid_features,
                             #y: valid_labels,
                             keep_prob: 1.
                         })
    
    test_acc, summary_test = sess.run([accuracy_test,merged_op], feed_dict={x: batch_features_test, y: batch_labels_test, keep_prob: 1.})
    
    train_writer.add_summary(summary_train, 1)
    test_writer.add_summary(summary_test, count)
    sum_writer.add_summary(summary_test, count)
    
    count += 1
    
    print('Loss: {:>2.4f} , Training Accuracy: {:>2.6f} , Testing Accuracy: {:>2.6f}'.format(loss, valid_acc, test_acc))


# ### Fully Train the Model

# In[ ]:

def labelAugment(batch_labels):
    augmLabels=[]
    for x in batch_labels:
        augmLabels.append(x)
        augmLabels.append(x)
        augmLabels.append(x)
    return np.array(augmLabels)

save_model_path = './image_classification'

tf.summary.FileWriterCache.clear()

epochs = 10
increment = 0
subbatch_size = 50
n_batches = 5

num_steps = 200     # 20 #loop over data once

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    
    sess.run(tf.global_variables_initializer())
    
    summary_writer = tf.summary.FileWriter(LOGDIR+'Q2', graph=tf.get_default_graph())
    
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        #n_batches = 5
        for batch_i in range(n_batches):
            print('Here')
            increment = 0
            for sub_batch in range(num_steps):
                batch_features = c10_train_dataset[batch_i][increment : (subbatch_size + increment)]
                batch_labels = c10_train_labels[batch_i][increment : (subbatch_size + increment)]
            
                batch_features_test = c10_test_dataset[increment : (subbatch_size + increment)]
                batch_labels_test = c10_test_labels[increment : (subbatch_size + increment)]
                
                augmented_batch_features=flipped_images = flip_images(batch_features)
                augmented_batch_features=np.vstack([batch_features,augmented_batch_features])
                
                augmentedLabels=labelAugment(batch_labels)
                augmentedLabels=np.vstack([batch_labels,augmentedLabels])
                train_neural_network(sess, optimizer, keep_probability, augmented_batch_features, augmentedLabels)
                                                                                        
                print('Epoch # {}, CIFAR-10 Batch # {}, chunk = [{}:{}]  '.format(epoch + 1, batch_i,
                                                                                   increment, (subbatch_size + increment)), end='')
                increment += subbatch_size
                
                print_stats(sess, augmented_batch_features, augmentedLabels, cost, batch_features_test,batch_labels_test, merged_summary_op, summary_writer)
                

