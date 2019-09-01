---
layout: page
title: ANN Basics - MNIST Numbers
permalink: /project_4/
---



### Purpose:

This notebook is an informal set of personal notes on the implementation of Neural Nets in TensorFlow.  This content primarily follows content from chapters 10-13 of "__Hands-On Machine Learning with Scikit_Learn & TensorFlow__" by Aurelien Geron and chapters 7 & 8 of "__Deep Learning__" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville

### Table of Contents:

- Tensorflow Setup
- Model Construction
    + Setup
    + Layer Definition
    + Loss & Training
    + Model Log
    + Execute Training
- Visualization in TensorBoard
- Batch Normalization
- Deep Dive: Regularization
    + Parameter Norm Penalties
    + Early Stopping
    + Dropout
    + Data Augmentation
- Next Steps


```python
#Notebook Setup
# Common imports
import numpy as np
import os

# To plot pretty figures
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
```

### TensorFlow Setup

TensorFlow graphs need to be reset between construction phases within a notebook (or manage multiple 'graphs' by switching the default graph).  Also, need to double check that the GPU is recognized.


```python
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
import tensorflow as tf
tf.test.gpu_device_name()
#should output: '/device:GPU:0' if one GPU is recognized and working properly
```




    ''



### Model Construction

Neural Net construction in TensorFlow can be broken into several phases:

| Construction Phase        | Steps           |
| ------------- | ------------- |
| Setup      | Define inputs, setup TensorBoard directory, etc. |
| Layer Definition    | Define and connect Net layers      |
| Loss & Training Functions | Define Loss function and Training functions |
| Model Log | Collect TensorBoard Summaries |
| Train | Execute the training operation |

#### Model Setup:


```python
#Always reset graph (unless managing multiple graphs)
reset_graph()

#setup naming convention for TensorBoard logs
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/ANN_Basics_run-{}".format(root_logdir, now)

he_init = tf.contrib.layers.variance_scaling_initializer()

#setup Neural Net Layer function with Tensorboard hooks
def fc_layer(input, channels_in, channels_out, name):
    with tf.name_scope(name):
        w = tf.Variable(he_init([channels_in, channels_out]), name="W") #random values to initialize weights
        b = tf.Variable(tf.zeros([channels_out]), name="B")
        act = tf.nn.relu(tf.matmul(input,w) + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act
    
def logit_layer(input, channels_in, channels_out, name):
    with tf.name_scope(name):
        w = tf.Variable(he_init([channels_in, channels_out]), name="W") #random values to initialize weights
        b = tf.Variable(tf.zeros([channels_out]), name="B")
        act = tf.matmul(input,w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

#Setup input variable definitions
#MNIST numbers example
height = 28
width = 28
channels = 1
n_inputs = height * width


```

#### Layer Definition:


```python
#define layer sizes
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

#define placeholders for the training data set
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

#define model layers
with tf.name_scope("DNN"):                        #name_scope helps orgainize and clarify TensorBoard graph
    hidden1 = fc_layer(X, n_inputs, n_hidden1, name="hidden1")
    hidden2 = fc_layer(hidden1, n_hidden1, n_hidden2, name="hidden2")
    logits = logit_layer(hidden2, n_hidden2, n_outputs, name="logits")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")
```

#### Loss & Training Functions:


```python
#define loss function
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
#tf.summary.scalar('cross_entropy', xentropy)

#define training operation
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

#define measure of accuracy to track
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
accuracy_summary = tf.summary.scalar('accuracy', accuracy)
```

#### Model Log:


```python
#Create initialization function
init = tf.global_variables_initializer()

#Initialize model saver
saver = tf.train.Saver()

#Initialize TensorBoard file writer and collect all data hooks
#file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
#tf.summary.image('input', X_reshaped, 3)
merged_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(logdir + 'train/', tf.get_default_graph())
test_writer = tf.summary.FileWriter(logdir + 'test/')
```

#### Training: 


```python
import time
import input_data
mnist = input_data.read_data_sets("MNIST_data/")

n_epochs = 40
batch_size = 20

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        start = time.time()
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y:y_batch})
            if iteration % 10 == 0:
                summary_str = sess.run(merged_summary, feed_dict={X: X_batch, 
                                                          y: y_batch})
                step = epoch * mnist.train.num_examples // batch_size + iteration
                train_writer.add_summary(summary_str, step)
                summary_str, acc = sess.run([accuracy_summary, accuracy], feed_dict={X: mnist.test.images, y:mnist.test.labels})
                test_writer.add_summary(summary_str, step)
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y:mnist.test.labels})
        print(epoch, "Train accuracy:" , acc_train, "Test accuracy: ", acc_test)
        end = time.time()
        print("Epoch runtime:", (end-start))
        
    save_path = saver.save(sess, "./my_model_final.ckpt")
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
    0 Train accuracy: 0.95 Test accuracy:  0.9284
    Epoch runtime: 14.590498685836792
    1 Train accuracy: 1.0 Test accuracy:  0.9448
    Epoch runtime: 15.168869256973267
    2 Train accuracy: 1.0 Test accuracy:  0.9525
    Epoch runtime: 15.151599407196045
    3 Train accuracy: 0.95 Test accuracy:  0.9583
    Epoch runtime: 15.123012781143188
    4 Train accuracy: 0.95 Test accuracy:  0.9633
    Epoch runtime: 15.178619623184204
    5 Train accuracy: 1.0 Test accuracy:  0.9646
    Epoch runtime: 14.930349349975586
    6 Train accuracy: 1.0 Test accuracy:  0.9675
    Epoch runtime: 15.115957736968994
    7 Train accuracy: 0.95 Test accuracy:  0.9676
    Epoch runtime: 15.1262788772583
    8 Train accuracy: 1.0 Test accuracy:  0.9729
    Epoch runtime: 14.645480394363403
    9 Train accuracy: 1.0 Test accuracy:  0.974
    Epoch runtime: 14.992468118667603
    10 Train accuracy: 1.0 Test accuracy:  0.9726
    Epoch runtime: 14.679440021514893
    11 Train accuracy: 1.0 Test accuracy:  0.9745
    Epoch runtime: 15.435067176818848
    12 Train accuracy: 1.0 Test accuracy:  0.9763
    Epoch runtime: 15.039925336837769
    13 Train accuracy: 1.0 Test accuracy:  0.9737
    Epoch runtime: 14.521102905273438
    14 Train accuracy: 1.0 Test accuracy:  0.9767
    Epoch runtime: 14.222981452941895
    15 Train accuracy: 1.0 Test accuracy:  0.977
    Epoch runtime: 14.31706428527832
    16 Train accuracy: 1.0 Test accuracy:  0.9775
    Epoch runtime: 14.334947109222412
    17 Train accuracy: 1.0 Test accuracy:  0.9768
    Epoch runtime: 14.5609712600708
    18 Train accuracy: 1.0 Test accuracy:  0.9773
    Epoch runtime: 14.256910562515259
    19 Train accuracy: 1.0 Test accuracy:  0.9782
    Epoch runtime: 14.425873517990112
    20 Train accuracy: 1.0 Test accuracy:  0.9783
    Epoch runtime: 15.36442232131958
    21 Train accuracy: 1.0 Test accuracy:  0.9783
    Epoch runtime: 14.679144859313965
    22 Train accuracy: 1.0 Test accuracy:  0.9766
    Epoch runtime: 14.24965524673462
    23 Train accuracy: 1.0 Test accuracy:  0.9797
    Epoch runtime: 14.260687828063965
    24 Train accuracy: 1.0 Test accuracy:  0.9782
    Epoch runtime: 14.52859902381897
    25 Train accuracy: 1.0 Test accuracy:  0.979
    Epoch runtime: 14.430864095687866
    26 Train accuracy: 1.0 Test accuracy:  0.9798
    Epoch runtime: 14.488142490386963
    27 Train accuracy: 1.0 Test accuracy:  0.9797
    Epoch runtime: 14.304199457168579
    28 Train accuracy: 1.0 Test accuracy:  0.9788
    Epoch runtime: 14.348829507827759
    29 Train accuracy: 1.0 Test accuracy:  0.9797
    Epoch runtime: 14.374854803085327
    30 Train accuracy: 1.0 Test accuracy:  0.9797
    Epoch runtime: 14.392420530319214
    31 Train accuracy: 1.0 Test accuracy:  0.9795
    Epoch runtime: 14.566481113433838
    32 Train accuracy: 1.0 Test accuracy:  0.9801
    Epoch runtime: 14.627853631973267
    33 Train accuracy: 1.0 Test accuracy:  0.9789
    Epoch runtime: 14.683981895446777
    34 Train accuracy: 1.0 Test accuracy:  0.9802
    Epoch runtime: 14.568735361099243
    35 Train accuracy: 1.0 Test accuracy:  0.9796
    Epoch runtime: 14.37185025215149
    36 Train accuracy: 1.0 Test accuracy:  0.9801
    Epoch runtime: 14.383451461791992
    37 Train accuracy: 1.0 Test accuracy:  0.9807
    Epoch runtime: 14.472146987915039
    38 Train accuracy: 1.0 Test accuracy:  0.9804
    Epoch runtime: 14.376425504684448
    39 Train accuracy: 1.0 Test accuracy:  0.9801
    Epoch runtime: 14.376371622085571


### Visualization in TensorBoard

Open new terminal in project folder:

`$ tensorboard --logdir tf_logs/`

TensorBoard automatically generates a visualization of the model structure:

![]({{site.url}}/assets/Project4_ANN Basics_Files/TensorBoard_Graph.png)


Plot for the Accuracy measure logged with each iteration:

![]({{site.url}}/assets/Project4_ANN Basics_Files/TensorBoard_Accuracy.png)

Histogram plots for all weights, biases, and activation values for each layer for each iteration:

![]({{site.url}}/assets/Project4_ANN Basics_Files/TensorBoard_Histograms.png)




### Batch Normalization

Batch Normalization is a process of normalizing and shifting the distribution of inputs to the activation function in each layer.  This process minimizes the changes between batches during training and it controls the inputs to minimize the likelihood of a vanishing or exploding gradient for optimization.

__During Training:__

- The mean and standard deviation for each Batch of U = `tf.matmul(input,w) + b` is calculated
- The distribution of U for the batch is de-meaned and standard deviation set to 1
- U is scaled and shifted by two new 'learned' terms: gamma*U + beta
  + gamma and beta are learned along with the rest of the weights (w) and biases (b) as part of the optimization function
- The mean (mu) and standard deviation (sigma) for the whole training set is learned as well using a rolling updated average
  + this rolling value is an exponential decay rate that is set as a hyper-parameter
  
__During Inference:__

- Individual instances are de-meaned and normalized using the mean (mu) and standard deviation (sigma) from the whole training dataset
- The learned scaling and offset parameters, gamma and beta, are applied

In the example below, `batch_norm()` is used to handle all of the operations above.  

Note: `batch_norm()` by default has gamma value fixed at 1.  This is appropriate when `nn.relu()` is used as the activation function, or if the activation function is linear (the weights in the next layer do the scaling already).

Note: Note: When is_training is True the moving_mean and moving_variance need to be updated, by default the update_ops are placed in tf.GraphKeys.UPDATE_OPS so they need to be added as a dependency to the train_op.  when you execute an operation (such as train_step), only the subgraph components relevant to train_step will be executed. Unfortunately, the update_moving_averages operation is not a parent of train_step in the computational graph, so we will never update the moving averages! To account for this we need to add an extra operation to the `sess.run()` function: `tf.get_collection(tf.GraphKeys.UPDATE_OPS)`

Argument to control gamma:

scale: If True, multiply by gamma. If False, gamma is not used. When the next layer is linear (also e.g. nn.relu), this can be disabled since the scaling can be done by the next layer.

Argument to control collections updates:

updates_collections: Collections to collect the update ops for computation. The updates_ops need to be executed with the train_op. If None, a control dependency would be added to make sure the updates are computed in place.

#### Model Setup: Batch Normalization



```python
#Always reset graph (unless managing multiple graphs)
reset_graph()

from tensorflow.contrib.layers import batch_norm
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

#setup naming convention for TensorBoard logs
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/ANN_Basics_run-{}/".format(root_logdir, now)

he_init = tf.contrib.layers.variance_scaling_initializer()

#setup Neural Net Layer function with Tensorboard hooks
def fc_layer(input, channels_in, channels_out, name):
    with tf.name_scope(name):
        w = tf.Variable(he_init([channels_in, channels_out]), name="W") #random values to initialize weights
        b = tf.Variable(tf.zeros([channels_out]), name="B")
        U = tf.matmul(input,w)+b
        z = batch_norm(U, is_training=is_training, decay=0.9, updates_collections=None, scale=True)
        act = tf.nn.elu(z)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act
    
def logit_layer(input, channels_in, channels_out, name):
    with tf.name_scope(name):
        w = tf.Variable(he_init([channels_in, channels_out]), name="W") #random values to initialize weights
        b = tf.Variable(tf.zeros([channels_out]), name="B")
        U = tf.matmul(input,w) + b
        act = batch_norm(U, is_training=is_training, decay=0.9, updates_collections=None)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

#Setup input variable definitions
#MNIST numbers example
height = 28
width = 28
channels = 1
n_inputs = height * width
```

#### Layer Definition: Batch Normalization


```python
#define layer sizes
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

#define placeholders for the training data set
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

#define model layers
with tf.name_scope("DNN"):                        #name_scope helps orgainize and clarify TensorBoard graph
    hidden1 = fc_layer(X, n_inputs, n_hidden1, name="hidden1")
    hidden2 = fc_layer(hidden1, n_hidden1, n_hidden2, name="hidden2")
    logits = logit_layer(hidden2, n_hidden2, n_outputs, name="logits")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")
```

#### Loss & Training Functions: Batch Normalization


```python
#define loss function
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
#tf.summary.scalar('cross_entropy', xentropy)

#define training operation
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

#define measure of accuracy to track
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
accuracy_summary = tf.summary.scalar('accuracy', accuracy)
```

#### Model Log: Batch Normalization


```python
#Create initialization function
init = tf.global_variables_initializer()

#Initialize model saver
saver = tf.train.Saver()

#Initialize TensorBoard file writer and collect all data hooks
X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
#tf.summary.image('input', X_reshaped, 3)
merged_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(logdir + 'train/', tf.get_default_graph())
test_writer = tf.summary.FileWriter(logdir + 'test/')
```

#### Training: Batch Normalization


```python
import time
import input_data
mnist = input_data.read_data_sets("MNIST_data/")

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

n_epochs = 20
batch_size = 20

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        start = time.time()
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run([training_op, extra_update_ops], feed_dict={is_training: True, X: X_batch, y:y_batch})
            if iteration % 10 == 0:
                summary_str = sess.run(merged_summary, feed_dict={is_training: True, X: X_batch, 
                                                          y: y_batch})
                step = epoch * mnist.train.num_examples // batch_size + iteration
                train_writer.add_summary(summary_str, step)
                summary_str, acc = sess.run([accuracy_summary, accuracy], feed_dict={is_training: False,
                                                                                     X: mnist.test.images, 
                                                                                     y:mnist.test.labels})
                test_writer.add_summary(summary_str, step)
        acc_train = accuracy.eval(feed_dict={is_training: True, X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={is_training: False, X: mnist.test.images, y:mnist.test.labels})
        
        
        print(epoch, "Train accuracy:" , acc_train, "Test accuracy: ", acc_test)
        end = time.time()
        print("Epoch runtime:", (end-start))
        
    save_path = saver.save(sess, "./my_model_final.ckpt")
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
    0 Train accuracy: 0.95 Test accuracy:  0.9241
    Epoch runtime: 24.386589765548706
    1 Train accuracy: 0.95 Test accuracy:  0.9391
    Epoch runtime: 24.236037969589233
    2 Train accuracy: 0.95 Test accuracy:  0.9497
    Epoch runtime: 24.187593936920166
    3 Train accuracy: 0.85 Test accuracy:  0.954
    Epoch runtime: 24.295372009277344
    4 Train accuracy: 1.0 Test accuracy:  0.9582
    Epoch runtime: 24.849113702774048
    5 Train accuracy: 0.8 Test accuracy:  0.9565
    Epoch runtime: 24.30752992630005
    6 Train accuracy: 0.95 Test accuracy:  0.9626
    Epoch runtime: 24.253839254379272
    7 Train accuracy: 0.95 Test accuracy:  0.9649
    Epoch runtime: 24.142624855041504
    8 Train accuracy: 0.9 Test accuracy:  0.9627
    Epoch runtime: 24.231298685073853
    9 Train accuracy: 1.0 Test accuracy:  0.9658
    Epoch runtime: 24.142414093017578
    10 Train accuracy: 0.95 Test accuracy:  0.9662
    Epoch runtime: 24.10838508605957
    11 Train accuracy: 0.95 Test accuracy:  0.967
    Epoch runtime: 24.005409955978394
    12 Train accuracy: 1.0 Test accuracy:  0.9701
    Epoch runtime: 24.149910926818848
    13 Train accuracy: 1.0 Test accuracy:  0.9675
    Epoch runtime: 24.011756896972656
    14 Train accuracy: 0.95 Test accuracy:  0.9699
    Epoch runtime: 24.395891904830933
    15 Train accuracy: 1.0 Test accuracy:  0.9709
    Epoch runtime: 24.061220169067383
    16 Train accuracy: 0.9 Test accuracy:  0.971
    Epoch runtime: 24.681432485580444
    17 Train accuracy: 0.95 Test accuracy:  0.9708
    Epoch runtime: 24.569782733917236
    18 Train accuracy: 0.95 Test accuracy:  0.9718
    Epoch runtime: 24.641958236694336
    19 Train accuracy: 1.0 Test accuracy:  0.973
    Epoch runtime: 24.391478300094604


The model takes longer to train with Batch Normalization, but it reduces over-fitting (shown by the reduction in Train accuracy) and minimizes the problem of vanishing or exploding gradients.  So, over a larger series of epochs the model will eventually perform better than without the Batch Normalization.

![](./ANN Basics_Files/TensorBoard_BN.png)

![](./ANN Basics_Files/TensorBoard_BN_Legend.png)

### Deep Dive: Regularization

__Deep Learning:__ "any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error"

Regularization has 3 primary forms:

1) Penalizing the model for higher complexity
  + *L1, L2, etc.* penalities are added for Neural Network weights that are too strong or having too many weights
  + This forces the model to rely primarily on the most important inputs and not to specialize the model for rare instances
  + Early stopping also works to minimize complexity by preventing the model to continue training after an optimium generalization error is reached.
  
2) Adding Noise Robustness/ variance insensitivity
  + Adding random noise to weights (Noise Injection), Randomly dropping a portion of a neural layer (Dropout) during training add variance to the model to force it to not be too reliant on individual inputs.
  
3) More training data
  + Data Augmentation is used to increase the overall training dataset

#### L2 Regularization (Weight Decay or Ridge Regression)

L2 Regularization adds a term to the cost function that is tied to the total magnitude of the neural network weights (not the bias terms).

$ \ell' = \ell + \alpha \frac{1}{2} || w ||^2_2$

This has the effect of increasingly shrinking the weight vector on each step.  The $\alpha$ term is a hyper-parameter that must additionally be set (0.01 is typical).

#### Model Setup: L2 Regularization



```python
#Always reset graph (unless managing multiple graphs)
reset_graph()

from tensorflow.contrib.layers import batch_norm
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

#setup naming convention for TensorBoard logs
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/ANN_Basics_run-{}/".format(root_logdir, now)

he_init = tf.contrib.layers.variance_scaling_initializer()

#setup Neural Net Layer function with Tensorboard hooks
def fc_layer(input, channels_in, channels_out, name):
    with tf.name_scope(name):
        w = tf.Variable(he_init([channels_in, channels_out]), name="W") #random values to initialize weights
        b = tf.Variable(tf.zeros([channels_out]), name="B")
        U = tf.matmul(input,w)+b
        z = batch_norm(U, is_training=is_training, decay=0.9, updates_collections=None, scale=True)
        act = tf.nn.elu(z)
        W_mag = tf.reduce_sum(tf.matmul(tf.transpose(w),w))
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act, W_mag
    
def logit_layer(input, channels_in, channels_out, name):
    with tf.name_scope(name):
        w = tf.Variable(he_init([channels_in, channels_out]), name="W") #random values to initialize weights
        b = tf.Variable(tf.zeros([channels_out]), name="B")
        U = tf.matmul(input,w) + b
        act = batch_norm(U, is_training=is_training, decay=0.9, updates_collections=None)
        W_mag = tf.reduce_sum(tf.matmul(tf.transpose(w),w))
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act, W_mag

#Setup input variable definitions
#MNIST numbers example
height = 28
width = 28
channels = 1
n_inputs = height * width
```

#### Layer Definition: L2 Regularization


```python
#define layer sizes
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

#define placeholders for the training data set
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

#define model layers
with tf.name_scope("DNN"):                        #name_scope helps orgainize and clarify TensorBoard graph
    hidden1, hidden1_l2 = fc_layer(X, n_inputs, n_hidden1, name="hidden1")
    hidden2, hidden2_l2 = fc_layer(hidden1, n_hidden1, n_hidden2, name="hidden2")
    logits, logits_l2 = logit_layer(hidden2, n_hidden2, n_outputs, name="logits")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")
```

#### Loss & Training Functions: L2 Regularization


```python
scale = 0.01 # alpha L2 regularization hyper-parameter

#define loss function
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    base_loss = tf.reduce_mean(xentropy, name="xentropy")
    reg_losses = hidden1_l2 + hidden2_l2 + logits_l2
    loss = tf.add(base_loss, scale * reg_losses, name="loss")
#tf.summary.scalar('cross_entropy', xentropy)

#define training operation
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

#define measure of accuracy to track
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
accuracy_summary = tf.summary.scalar('accuracy', accuracy)
```

#### Model Log: L2 Regularization


```python
#Create initialization function
init = tf.global_variables_initializer()

#Initialize model saver
saver = tf.train.Saver()

#Initialize TensorBoard file writer and collect all data hooks
X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
#tf.summary.image('input', X_reshaped, 3)
merged_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(logdir + 'train/', tf.get_default_graph())
test_writer = tf.summary.FileWriter(logdir + 'test/')
```

#### Training: L2 Regularization


```python
import time
import input_data
mnist = input_data.read_data_sets("MNIST_data/")

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

n_epochs = 20
batch_size = 20

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        start = time.time()
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run([training_op, extra_update_ops], feed_dict={is_training: True, X: X_batch, y:y_batch})
            if iteration % 10 == 0:
                summary_str = sess.run(merged_summary, feed_dict={is_training: True, X: X_batch, 
                                                          y: y_batch})
                step = epoch * mnist.train.num_examples // batch_size + iteration
                train_writer.add_summary(summary_str, step)
                summary_str, acc = sess.run([accuracy_summary, accuracy], feed_dict={is_training: False,
                                                                                     X: mnist.test.images, 
                                                                                     y:mnist.test.labels})
                test_writer.add_summary(summary_str, step)
        acc_train = accuracy.eval(feed_dict={is_training: True, X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={is_training: False, X: mnist.test.images, y:mnist.test.labels})
        
        
        print(epoch, "Train accuracy:" , acc_train, "Test accuracy: ", acc_test)
        end = time.time()
        print("Epoch runtime:", (end-start))
        
    save_path = saver.save(sess, "./my_model_final.ckpt")
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
    0 Train accuracy: 0.95 Test accuracy:  0.929
    Epoch runtime: 31.215134143829346
    1 Train accuracy: 0.95 Test accuracy:  0.9447
    Epoch runtime: 30.870367288589478
    2 Train accuracy: 0.95 Test accuracy:  0.9526
    Epoch runtime: 30.745681762695312
    3 Train accuracy: 0.85 Test accuracy:  0.9566
    Epoch runtime: 30.504931211471558
    4 Train accuracy: 0.95 Test accuracy:  0.96
    Epoch runtime: 30.3809175491333
    5 Train accuracy: 0.85 Test accuracy:  0.957
    Epoch runtime: 30.411189794540405
    6 Train accuracy: 0.95 Test accuracy:  0.9643
    Epoch runtime: 30.37272620201111
    7 Train accuracy: 0.9 Test accuracy:  0.9663
    Epoch runtime: 30.322518587112427
    8 Train accuracy: 0.9 Test accuracy:  0.9665
    Epoch runtime: 30.49232244491577
    9 Train accuracy: 1.0 Test accuracy:  0.9695
    Epoch runtime: 30.799328327178955
    10 Train accuracy: 1.0 Test accuracy:  0.9685
    Epoch runtime: 30.758010387420654
    11 Train accuracy: 0.95 Test accuracy:  0.9689
    Epoch runtime: 30.53590488433838
    12 Train accuracy: 1.0 Test accuracy:  0.9715
    Epoch runtime: 30.545585870742798
    13 Train accuracy: 1.0 Test accuracy:  0.9685
    Epoch runtime: 30.98749279975891
    14 Train accuracy: 0.95 Test accuracy:  0.9719
    Epoch runtime: 30.4417941570282
    15 Train accuracy: 0.95 Test accuracy:  0.9726
    Epoch runtime: 30.196251392364502
    16 Train accuracy: 0.9 Test accuracy:  0.9706
    Epoch runtime: 30.182875394821167
    17 Train accuracy: 0.85 Test accuracy:  0.9711
    Epoch runtime: 30.146605968475342
    18 Train accuracy: 0.9 Test accuracy:  0.9716
    Epoch runtime: 30.191405773162842
    19 Train accuracy: 1.0 Test accuracy:  0.9726
    Epoch runtime: 30.156668186187744


#### L1 Regularization (Least Absolute Shrinkage and Selection Operator)

L1 Regularization adds a term to the cost function that is tied to the L1 norm of the neural network weights (not the bias terms).

$ \ell' = \ell + \alpha  || w ||_1$

This has the effect of zeroing out (or nearly zeroing) the weights of the least important features.  The $\alpha$ term is a hyper-parameter that must additionally be set (0.01 is typical).

#### Model Setup: L1 Regularization



```python
#Always reset graph (unless managing multiple graphs)
reset_graph()

from tensorflow.contrib.layers import batch_norm
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

#setup naming convention for TensorBoard logs
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/ANN_Basics_run-{}/".format(root_logdir, now)

he_init = tf.contrib.layers.variance_scaling_initializer()

#setup Neural Net Layer function with Tensorboard hooks
def fc_layer(input, channels_in, channels_out, name):
    with tf.name_scope(name):
        w = tf.Variable(he_init([channels_in, channels_out]), name="W") #random values to initialize weights
        b = tf.Variable(tf.zeros([channels_out]), name="B")
        U = tf.matmul(input,w)+b
        z = batch_norm(U, is_training=is_training, decay=0.9, updates_collections=None, scale=True)
        act = tf.nn.elu(z)
        W_mag = tf.reduce_sum(tf.abs(w))
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act, W_mag
    
def logit_layer(input, channels_in, channels_out, name):
    with tf.name_scope(name):
        w = tf.Variable(he_init([channels_in, channels_out]), name="W") #random values to initialize weights
        b = tf.Variable(tf.zeros([channels_out]), name="B")
        U = tf.matmul(input,w) + b
        act = batch_norm(U, is_training=is_training, decay=0.9, updates_collections=None)
        W_mag = tf.reduce_sum(tf.abs(w))
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act, W_mag

#Setup input variable definitions
#MNIST numbers example
height = 28
width = 28
channels = 1
n_inputs = height * width
```

#### Layer Definition: L1 Regularization


```python
#define layer sizes
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

#define placeholders for the training data set
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

#define model layers
with tf.name_scope("DNN"):                        #name_scope helps orgainize and clarify TensorBoard graph
    hidden1, hidden1_l1 = fc_layer(X, n_inputs, n_hidden1, name="hidden1")
    hidden2, hidden2_l1 = fc_layer(hidden1, n_hidden1, n_hidden2, name="hidden2")
    logits, logits_l1 = logit_layer(hidden2, n_hidden2, n_outputs, name="logits")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")
```

#### Loss & Training Functions: L1 Regularization


```python
scale = 0.01 # alpha L1 regularization hyper-parameter

#define loss function
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    base_loss = tf.reduce_mean(xentropy, name="xentropy")
    reg_losses = hidden1_l1 + hidden2_l1 + logits_l1
    loss = tf.add(base_loss, scale * reg_losses, name="loss")
#tf.summary.scalar('cross_entropy', xentropy)

#define training operation
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

#define measure of accuracy to track
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
accuracy_summary = tf.summary.scalar('accuracy', accuracy)
```

#### Model Log: L1 Regularization


```python
#Create initialization function
init = tf.global_variables_initializer()

#Initialize model saver
saver = tf.train.Saver()

#Initialize TensorBoard file writer and collect all data hooks
X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
#tf.summary.image('input', X_reshaped, 3)
merged_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(logdir + 'train/', tf.get_default_graph())
test_writer = tf.summary.FileWriter(logdir + 'test/')
```

#### Training: L1 Regularization


```python
import time
import input_data
mnist = input_data.read_data_sets("MNIST_data/")

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

n_epochs = 20
batch_size = 20

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        start = time.time()
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run([training_op, extra_update_ops], feed_dict={is_training: True, X: X_batch, y:y_batch})
            if iteration % 10 == 0:
                summary_str = sess.run(merged_summary, feed_dict={is_training: True, X: X_batch, 
                                                          y: y_batch})
                step = epoch * mnist.train.num_examples // batch_size + iteration
                train_writer.add_summary(summary_str, step)
                summary_str, acc = sess.run([accuracy_summary, accuracy], feed_dict={is_training: False,
                                                                                     X: mnist.test.images, 
                                                                                     y:mnist.test.labels})
                test_writer.add_summary(summary_str, step)
        acc_train = accuracy.eval(feed_dict={is_training: True, X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={is_training: False, X: mnist.test.images, y:mnist.test.labels})
        
        
        print(epoch, "Train accuracy:" , acc_train, "Test accuracy: ", acc_test)
        end = time.time()
        print("Epoch runtime:", (end-start))
        
    save_path = saver.save(sess, "./my_model_final.ckpt")
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
    0 Train accuracy: 1.0 Test accuracy:  0.8445
    Epoch runtime: 28.35469365119934
    1 Train accuracy: 1.0 Test accuracy:  0.8616
    Epoch runtime: 27.769091606140137
    2 Train accuracy: 0.95 Test accuracy:  0.817
    Epoch runtime: 27.973952293395996
    3 Train accuracy: 0.9 Test accuracy:  0.8473
    Epoch runtime: 27.860862970352173
    4 Train accuracy: 1.0 Test accuracy:  0.8681
    Epoch runtime: 28.112948894500732
    5 Train accuracy: 0.95 Test accuracy:  0.8218
    Epoch runtime: 27.917056798934937
    6 Train accuracy: 0.9 Test accuracy:  0.8672
    Epoch runtime: 28.828511714935303
    7 Train accuracy: 1.0 Test accuracy:  0.8734
    Epoch runtime: 28.415383100509644
    8 Train accuracy: 0.85 Test accuracy:  0.834
    Epoch runtime: 28.546268939971924
    9 Train accuracy: 1.0 Test accuracy:  0.8512
    Epoch runtime: 28.387388944625854
    10 Train accuracy: 0.95 Test accuracy:  0.8855
    Epoch runtime: 28.89906907081604
    11 Train accuracy: 1.0 Test accuracy:  0.8409
    Epoch runtime: 28.539801359176636
    12 Train accuracy: 0.9 Test accuracy:  0.8289
    Epoch runtime: 28.42889165878296
    13 Train accuracy: 0.9 Test accuracy:  0.8448
    Epoch runtime: 28.64804768562317
    14 Train accuracy: 0.9 Test accuracy:  0.8414
    Epoch runtime: 28.86787223815918
    15 Train accuracy: 1.0 Test accuracy:  0.8932
    Epoch runtime: 28.296983003616333
    16 Train accuracy: 0.85 Test accuracy:  0.8649
    Epoch runtime: 28.77966856956482
    17 Train accuracy: 0.95 Test accuracy:  0.8702
    Epoch runtime: 28.36933207511902
    18 Train accuracy: 0.9 Test accuracy:  0.8753
    Epoch runtime: 28.159512281417847
    19 Train accuracy: 0.95 Test accuracy:  0.9019
    Epoch runtime: 28.401350736618042


#### Elastic Net (Combination of Ridge Regression and LASSO)

Elastic Net Regularization adds both an L1 and an L2 term to the cost function that is tied to the neural network weights (not the bias terms).

$ \ell' = \ell + r \alpha  || w ||_1 + \frac{1-r}{2} \alpha || w ||^2_2$

This balances the effect between Ridge Regression and LASSO based on the $r$ term.  If $r$ is equal to zero, then this is equivalent to using L2 norm alone, and if $r$ is equal to 1, then this is equivalent to using L1 norm alone.  The $\alpha$ term is a hyper-parameter that must additionally be set (0.01 is typical).

#### Model Setup: Elastic Net



```python
#Always reset graph (unless managing multiple graphs)
reset_graph()

from tensorflow.contrib.layers import batch_norm
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

#setup naming convention for TensorBoard logs
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/ANN_Basics_run-{}/".format(root_logdir, now)

he_init = tf.contrib.layers.variance_scaling_initializer()

#setup Neural Net Layer function with Tensorboard hooks
def fc_layer(input, channels_in, channels_out, name):
    with tf.name_scope(name):
        w = tf.Variable(he_init([channels_in, channels_out]), name="W") #random values to initialize weights
        b = tf.Variable(tf.zeros([channels_out]), name="B")
        U = tf.matmul(input,w)+b
        z = batch_norm(U, is_training=is_training, decay=0.9, updates_collections=None, scale=True)
        act = tf.nn.elu(z)
        W_L1 = tf.reduce_sum(tf.abs(w))
        W_L2 = tf.reduce_sum(tf.matmul(tf.transpose(w),w))
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act, W_L1, W_L2
    
def logit_layer(input, channels_in, channels_out, name):
    with tf.name_scope(name):
        w = tf.Variable(he_init([channels_in, channels_out]), name="W") #random values to initialize weights
        b = tf.Variable(tf.zeros([channels_out]), name="B")
        U = tf.matmul(input,w) + b
        act = batch_norm(U, is_training=is_training, decay=0.9, updates_collections=None)
        W_L1 = tf.reduce_sum(tf.abs(w))
        W_L2 = tf.reduce_sum(tf.matmul(tf.transpose(w),w))
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act, W_L1, W_L2

#Setup input variable definitions
#MNIST numbers example
height = 28
width = 28
channels = 1
n_inputs = height * width
```

#### Layer Definition: Elastic Net


```python
#define layer sizes
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

#define placeholders for the training data set
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

#define model layers
with tf.name_scope("DNN"):                        #name_scope helps orgainize and clarify TensorBoard graph
    hidden1, hidden1_l1, hidden1_l2 = fc_layer(X, n_inputs, n_hidden1, name="hidden1")
    hidden2, hidden2_l1, hidden2_l2 = fc_layer(hidden1, n_hidden1, n_hidden2, name="hidden2")
    logits, logits_l1, logits_l2 = logit_layer(hidden2, n_hidden2, n_outputs, name="logits")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")
```

#### Loss & Training Functions: Elastic Net


```python
scale = 0.01 # alpha L1 regularization hyper-parameter
r = 0.2

#define loss function
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    base_loss = tf.reduce_mean(xentropy, name="xentropy")
    l1_losses = hidden1_l1 + hidden2_l1 + logits_l1
    l2_losses = hidden1_l2 + hidden2_l2 + logits_l2
    reg_losses = r * l1_losses + ((1-r)/2) * l2_losses
    loss = tf.add(base_loss, scale * reg_losses, name="loss")
#tf.summary.scalar('cross_entropy', xentropy)

#define training operation
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

#define measure of accuracy to track
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
accuracy_summary = tf.summary.scalar('accuracy', accuracy)
```

#### Model Log: Elastic Net


```python
#Create initialization function
init = tf.global_variables_initializer()

#Initialize model saver
saver = tf.train.Saver()

#Initialize TensorBoard file writer and collect all data hooks
X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
#tf.summary.image('input', X_reshaped, 3)
merged_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(logdir + 'train/', tf.get_default_graph())
test_writer = tf.summary.FileWriter(logdir + 'test/')
```

#### Training: Elastic Net


```python
import time
import input_data
mnist = input_data.read_data_sets("MNIST_data/")

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

n_epochs = 20
batch_size = 20

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        start = time.time()
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run([training_op, extra_update_ops], feed_dict={is_training: True, X: X_batch, y:y_batch})
            if iteration % 10 == 0:
                summary_str = sess.run(merged_summary, feed_dict={is_training: True, X: X_batch, 
                                                          y: y_batch})
                step = epoch * mnist.train.num_examples // batch_size + iteration
                train_writer.add_summary(summary_str, step)
                summary_str, acc = sess.run([accuracy_summary, accuracy], feed_dict={is_training: False,
                                                                                     X: mnist.test.images, 
                                                                                     y:mnist.test.labels})
                test_writer.add_summary(summary_str, step)
        acc_train = accuracy.eval(feed_dict={is_training: True, X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={is_training: False, X: mnist.test.images, y:mnist.test.labels})
        
        
        print(epoch, "Train accuracy:" , acc_train, "Test accuracy: ", acc_test)
        end = time.time()
        print("Epoch runtime:", (end-start))
        
    save_path = saver.save(sess, "./my_model_final.ckpt")
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
    0 Train accuracy: 1.0 Test accuracy:  0.9245
    Epoch runtime: 36.0091495513916
    1 Train accuracy: 1.0 Test accuracy:  0.8994
    Epoch runtime: 35.533520221710205
    2 Train accuracy: 0.95 Test accuracy:  0.8722
    Epoch runtime: 35.48987007141113
    3 Train accuracy: 1.0 Test accuracy:  0.9185
    Epoch runtime: 35.57891607284546
    4 Train accuracy: 1.0 Test accuracy:  0.9302
    Epoch runtime: 35.50733041763306
    5 Train accuracy: 1.0 Test accuracy:  0.9284
    Epoch runtime: 35.01816940307617
    6 Train accuracy: 0.95 Test accuracy:  0.9229
    Epoch runtime: 35.191999435424805
    7 Train accuracy: 1.0 Test accuracy:  0.9309
    Epoch runtime: 35.23454403877258
    8 Train accuracy: 0.95 Test accuracy:  0.9212
    Epoch runtime: 35.19815397262573
    9 Train accuracy: 1.0 Test accuracy:  0.9216
    Epoch runtime: 35.35411834716797
    10 Train accuracy: 1.0 Test accuracy:  0.9381
    Epoch runtime: 35.25079417228699
    11 Train accuracy: 1.0 Test accuracy:  0.9186
    Epoch runtime: 35.23978877067566
    12 Train accuracy: 1.0 Test accuracy:  0.9286
    Epoch runtime: 35.3883535861969
    13 Train accuracy: 1.0 Test accuracy:  0.9267
    Epoch runtime: 35.230255365371704
    14 Train accuracy: 0.95 Test accuracy:  0.9325
    Epoch runtime: 35.75219106674194
    15 Train accuracy: 1.0 Test accuracy:  0.9427
    Epoch runtime: 35.204952001571655
    16 Train accuracy: 1.0 Test accuracy:  0.9362
    Epoch runtime: 34.990816593170166
    17 Train accuracy: 0.95 Test accuracy:  0.9339
    Epoch runtime: 35.22327375411987
    18 Train accuracy: 0.95 Test accuracy:  0.9394
    Epoch runtime: 35.01524305343628
    19 Train accuracy: 1.0 Test accuracy:  0.9398
    Epoch runtime: 35.72379660606384


### Early Stopping

__Deep Learning section 7.8__: *paraphrased* 'when training large models with the capacity to overfit the test data it is common to see training error steadily decrease over time, but for test error to reach a minimum early on in the training and then creep back up to a higher error rate.  This is indicative of over-fitting and is an example where early-stopping can yield a better overall model'

At typical approach to early stopping is to periodically evaluate the model on the test set, capture a snap-shot of the model and then only replace the snap-shot on the next test set if the validation score is improved.  That way the best version of the trained model on the test data is always kept in memory while training proceeds.

#### Model Setup: Early Stopping



```python
#Always reset graph (unless managing multiple graphs)
reset_graph()

from tensorflow.contrib.layers import batch_norm
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

#setup naming convention for TensorBoard logs
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/ANN_Basics_run-{}/".format(root_logdir, now)

he_init = tf.contrib.layers.variance_scaling_initializer()

#setup Neural Net Layer function with Tensorboard hooks
def fc_layer(input, channels_in, channels_out, name):
    with tf.name_scope(name):
        w = tf.Variable(he_init([channels_in, channels_out]), name="W") #random values to initialize weights
        b = tf.Variable(tf.zeros([channels_out]), name="B")
        U = tf.matmul(input,w)+b
        z = batch_norm(U, is_training=is_training, decay=0.9, updates_collections=None, scale=True)
        act = tf.nn.elu(z)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act
    
def logit_layer(input, channels_in, channels_out, name):
    with tf.name_scope(name):
        w = tf.Variable(he_init([channels_in, channels_out]), name="W") #random values to initialize weights
        b = tf.Variable(tf.zeros([channels_out]), name="B")
        U = tf.matmul(input,w) + b
        act = batch_norm(U, is_training=is_training, decay=0.9, updates_collections=None)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

#Setup input variable definitions
#MNIST numbers example
height = 28
width = 28
channels = 1
n_inputs = height * width
```

#### Layer Definition: Early Stopping


```python
#define layer sizes
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

#define placeholders for the training data set
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

#define model layers
with tf.name_scope("DNN"):                        #name_scope helps orgainize and clarify TensorBoard graph
    hidden1 = fc_layer(X, n_inputs, n_hidden1, name="hidden1")
    hidden2 = fc_layer(hidden1, n_hidden1, n_hidden2, name="hidden2")
    logits = logit_layer(hidden2, n_hidden2, n_outputs, name="logits")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")
```

#### Loss & Training Functions: Early Stopping


```python
#define loss function
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
#tf.summary.scalar('cross_entropy', xentropy)

#define training operation
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

#define measure of accuracy to track
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
accuracy_summary = tf.summary.scalar('accuracy', accuracy)
```

#### Model Log: Early Stopping


```python
#Create initialization function
init = tf.global_variables_initializer()

#Initialize model saver
saver = tf.train.Saver()

#Initialize TensorBoard file writer and collect all data hooks
X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
#tf.summary.image('input', X_reshaped, 3)
merged_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(logdir + 'train/', tf.get_default_graph())
test_writer = tf.summary.FileWriter(logdir + 'test/')
```

#### Training: Early Stopping


```python
import time
import input_data
mnist = input_data.read_data_sets("MNIST_data/")

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

n_epochs = 150
batch_size = 20

best_test = 0
best_epoch = 0

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        start = time.time()
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run([training_op, extra_update_ops], feed_dict={is_training: True, X: X_batch, y:y_batch})
            if iteration % 10 == 0:
                summary_str = sess.run(merged_summary, feed_dict={is_training: True, X: X_batch, 
                                                          y: y_batch})
                step = epoch * mnist.train.num_examples // batch_size + iteration
                train_writer.add_summary(summary_str, step)
                summary_str, acc = sess.run([accuracy_summary, accuracy], feed_dict={is_training: False,
                                                                                     X: mnist.test.images, 
                                                                                     y:mnist.test.labels})
                test_writer.add_summary(summary_str, step)
        acc_train = accuracy.eval(feed_dict={is_training: True, X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={is_training: False, X: mnist.test.images, y:mnist.test.labels})
        if acc_test > best_test:
            best_test = acc_test
            best_epoch = epoch
            save_path = saver.save(sess, "./best_early_stop.ckpt")
        if epoch > (best_epoch + 25):
            break
            
        print(epoch, "Train accuracy:" , acc_train, "Test accuracy: ", acc_test)
        end = time.time()
        print("Epoch runtime:", (end-start))
        
print('Best Epoch:', best_epoch)
print('Best Test:', best_test)
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
    0 Train accuracy: 0.95 Test accuracy:  0.9241
    Epoch runtime: 26.20693850517273
    1 Train accuracy: 0.95 Test accuracy:  0.9391
    Epoch runtime: 24.86277198791504
    2 Train accuracy: 0.95 Test accuracy:  0.9497
    Epoch runtime: 24.788996696472168
    3 Train accuracy: 0.85 Test accuracy:  0.954
    Epoch runtime: 24.913816213607788
    4 Train accuracy: 1.0 Test accuracy:  0.9582
    Epoch runtime: 24.357863903045654
    5 Train accuracy: 0.8 Test accuracy:  0.9565
    Epoch runtime: 24.524713039398193
    6 Train accuracy: 0.95 Test accuracy:  0.9626
    Epoch runtime: 24.659019708633423
    7 Train accuracy: 0.95 Test accuracy:  0.9649
    Epoch runtime: 24.852537870407104
    8 Train accuracy: 0.9 Test accuracy:  0.9627
    Epoch runtime: 25.84475326538086
    9 Train accuracy: 1.0 Test accuracy:  0.9658
    Epoch runtime: 24.1432466506958
    10 Train accuracy: 0.95 Test accuracy:  0.9662
    Epoch runtime: 23.797587156295776
    11 Train accuracy: 0.95 Test accuracy:  0.967
    Epoch runtime: 23.806743621826172
    12 Train accuracy: 1.0 Test accuracy:  0.9701
    Epoch runtime: 23.801499843597412
    13 Train accuracy: 1.0 Test accuracy:  0.9675
    Epoch runtime: 23.856434106826782
    14 Train accuracy: 0.95 Test accuracy:  0.9699
    Epoch runtime: 23.796069622039795
    15 Train accuracy: 1.0 Test accuracy:  0.9709
    Epoch runtime: 23.810758113861084
    16 Train accuracy: 0.9 Test accuracy:  0.971
    Epoch runtime: 23.796290397644043
    17 Train accuracy: 0.95 Test accuracy:  0.9708
    Epoch runtime: 23.764060735702515
    18 Train accuracy: 0.95 Test accuracy:  0.9718
    Epoch runtime: 23.792809009552002
    19 Train accuracy: 1.0 Test accuracy:  0.973
    Epoch runtime: 23.787736415863037
    20 Train accuracy: 1.0 Test accuracy:  0.9718
    Epoch runtime: 23.755197286605835
    21 Train accuracy: 0.95 Test accuracy:  0.9733
    Epoch runtime: 23.773937463760376
    22 Train accuracy: 0.95 Test accuracy:  0.9728
    Epoch runtime: 23.9306321144104
    23 Train accuracy: 1.0 Test accuracy:  0.9761
    Epoch runtime: 23.782400369644165
    24 Train accuracy: 1.0 Test accuracy:  0.9732
    Epoch runtime: 23.830172300338745
    25 Train accuracy: 0.95 Test accuracy:  0.9746
    Epoch runtime: 23.816159963607788
    26 Train accuracy: 0.95 Test accuracy:  0.9762
    Epoch runtime: 23.826308727264404
    27 Train accuracy: 1.0 Test accuracy:  0.9736
    Epoch runtime: 23.777552127838135
    28 Train accuracy: 0.95 Test accuracy:  0.9762
    Epoch runtime: 23.75916028022766
    29 Train accuracy: 0.85 Test accuracy:  0.9758
    Epoch runtime: 23.75330948829651
    30 Train accuracy: 1.0 Test accuracy:  0.9771
    Epoch runtime: 23.778809785842896
    31 Train accuracy: 1.0 Test accuracy:  0.9756
    Epoch runtime: 23.76627779006958
    32 Train accuracy: 1.0 Test accuracy:  0.9755
    Epoch runtime: 23.76955509185791
    33 Train accuracy: 0.95 Test accuracy:  0.9763
    Epoch runtime: 23.779017686843872
    34 Train accuracy: 0.95 Test accuracy:  0.9757
    Epoch runtime: 23.911543130874634
    35 Train accuracy: 0.95 Test accuracy:  0.9759
    Epoch runtime: 23.858562231063843
    36 Train accuracy: 1.0 Test accuracy:  0.9758
    Epoch runtime: 23.79735517501831
    37 Train accuracy: 0.95 Test accuracy:  0.9766
    Epoch runtime: 23.774149656295776
    38 Train accuracy: 1.0 Test accuracy:  0.9773
    Epoch runtime: 23.833223581314087
    39 Train accuracy: 0.85 Test accuracy:  0.977
    Epoch runtime: 23.752622365951538
    40 Train accuracy: 0.95 Test accuracy:  0.9772
    Epoch runtime: 23.978296279907227
    41 Train accuracy: 0.9 Test accuracy:  0.9764
    Epoch runtime: 23.734403371810913
    42 Train accuracy: 0.95 Test accuracy:  0.977
    Epoch runtime: 23.757582902908325
    43 Train accuracy: 1.0 Test accuracy:  0.9757
    Epoch runtime: 23.7353732585907
    44 Train accuracy: 1.0 Test accuracy:  0.9778
    Epoch runtime: 23.834320545196533
    45 Train accuracy: 1.0 Test accuracy:  0.9781
    Epoch runtime: 23.780799388885498
    46 Train accuracy: 0.9 Test accuracy:  0.9791
    Epoch runtime: 23.861382722854614
    47 Train accuracy: 0.95 Test accuracy:  0.9778
    Epoch runtime: 23.779444217681885
    48 Train accuracy: 0.95 Test accuracy:  0.9776
    Epoch runtime: 23.74011254310608
    49 Train accuracy: 0.95 Test accuracy:  0.9777
    Epoch runtime: 23.773831605911255
    50 Train accuracy: 1.0 Test accuracy:  0.9775
    Epoch runtime: 23.7661235332489
    51 Train accuracy: 1.0 Test accuracy:  0.9785
    Epoch runtime: 23.78847599029541
    52 Train accuracy: 0.95 Test accuracy:  0.9791
    Epoch runtime: 23.73590898513794
    53 Train accuracy: 0.9 Test accuracy:  0.9792
    Epoch runtime: 23.801321983337402
    54 Train accuracy: 0.95 Test accuracy:  0.9796
    Epoch runtime: 23.81420063972473
    55 Train accuracy: 1.0 Test accuracy:  0.9802
    Epoch runtime: 23.739454984664917
    56 Train accuracy: 0.9 Test accuracy:  0.9784
    Epoch runtime: 23.7623610496521
    57 Train accuracy: 1.0 Test accuracy:  0.9793
    Epoch runtime: 23.75600552558899
    58 Train accuracy: 0.95 Test accuracy:  0.9782
    Epoch runtime: 23.855237007141113
    59 Train accuracy: 0.95 Test accuracy:  0.9781
    Epoch runtime: 23.79854106903076
    60 Train accuracy: 1.0 Test accuracy:  0.979
    Epoch runtime: 23.73498034477234
    61 Train accuracy: 0.9 Test accuracy:  0.9805
    Epoch runtime: 23.827515840530396
    62 Train accuracy: 1.0 Test accuracy:  0.9791
    Epoch runtime: 23.782772541046143
    63 Train accuracy: 1.0 Test accuracy:  0.9799
    Epoch runtime: 23.786271333694458
    64 Train accuracy: 1.0 Test accuracy:  0.9803
    Epoch runtime: 23.8151752948761
    65 Train accuracy: 1.0 Test accuracy:  0.9811
    Epoch runtime: 24.089199542999268
    66 Train accuracy: 1.0 Test accuracy:  0.9808
    Epoch runtime: 23.775761604309082
    67 Train accuracy: 1.0 Test accuracy:  0.9799
    Epoch runtime: 23.791836738586426
    68 Train accuracy: 1.0 Test accuracy:  0.9811
    Epoch runtime: 23.78024673461914
    69 Train accuracy: 0.95 Test accuracy:  0.98
    Epoch runtime: 23.800148248672485
    70 Train accuracy: 1.0 Test accuracy:  0.9807
    Epoch runtime: 23.764111280441284
    71 Train accuracy: 1.0 Test accuracy:  0.981
    Epoch runtime: 23.814738512039185
    72 Train accuracy: 1.0 Test accuracy:  0.9812
    Epoch runtime: 23.819961071014404
    73 Train accuracy: 0.95 Test accuracy:  0.9823
    Epoch runtime: 23.807893753051758
    74 Train accuracy: 1.0 Test accuracy:  0.9801
    Epoch runtime: 23.79572629928589
    75 Train accuracy: 0.9 Test accuracy:  0.9805
    Epoch runtime: 23.75127124786377
    76 Train accuracy: 1.0 Test accuracy:  0.9813
    Epoch runtime: 23.791248083114624
    77 Train accuracy: 1.0 Test accuracy:  0.9823
    Epoch runtime: 23.840741395950317
    78 Train accuracy: 1.0 Test accuracy:  0.9814
    Epoch runtime: 23.772239208221436
    79 Train accuracy: 0.95 Test accuracy:  0.9817
    Epoch runtime: 23.860690593719482
    80 Train accuracy: 1.0 Test accuracy:  0.9817
    Epoch runtime: 23.821345806121826
    81 Train accuracy: 1.0 Test accuracy:  0.9812
    Epoch runtime: 23.80265736579895
    82 Train accuracy: 1.0 Test accuracy:  0.9817
    Epoch runtime: 23.78797197341919
    83 Train accuracy: 1.0 Test accuracy:  0.9819
    Epoch runtime: 23.811220407485962
    84 Train accuracy: 0.95 Test accuracy:  0.9814
    Epoch runtime: 23.800326824188232
    85 Train accuracy: 1.0 Test accuracy:  0.9826
    Epoch runtime: 23.801023483276367
    86 Train accuracy: 0.95 Test accuracy:  0.9812
    Epoch runtime: 23.761409282684326
    87 Train accuracy: 0.95 Test accuracy:  0.9821
    Epoch runtime: 23.805975914001465
    88 Train accuracy: 1.0 Test accuracy:  0.9816
    Epoch runtime: 23.775542974472046
    89 Train accuracy: 1.0 Test accuracy:  0.9826
    Epoch runtime: 23.83519148826599
    90 Train accuracy: 1.0 Test accuracy:  0.9824
    Epoch runtime: 23.780808687210083
    91 Train accuracy: 1.0 Test accuracy:  0.9824
    Epoch runtime: 23.95652985572815
    92 Train accuracy: 1.0 Test accuracy:  0.9821
    Epoch runtime: 23.79654812812805
    93 Train accuracy: 0.95 Test accuracy:  0.9824
    Epoch runtime: 23.880950212478638
    94 Train accuracy: 1.0 Test accuracy:  0.9827
    Epoch runtime: 23.851080417633057
    95 Train accuracy: 1.0 Test accuracy:  0.9817
    Epoch runtime: 23.837850093841553
    96 Train accuracy: 1.0 Test accuracy:  0.9833
    Epoch runtime: 23.840067625045776
    97 Train accuracy: 1.0 Test accuracy:  0.9819
    Epoch runtime: 23.838886737823486
    98 Train accuracy: 1.0 Test accuracy:  0.9816
    Epoch runtime: 23.795984983444214
    99 Train accuracy: 1.0 Test accuracy:  0.9825
    Epoch runtime: 23.814177989959717
    100 Train accuracy: 1.0 Test accuracy:  0.9826
    Epoch runtime: 23.796509265899658
    101 Train accuracy: 0.95 Test accuracy:  0.9823
    Epoch runtime: 23.76856231689453
    102 Train accuracy: 1.0 Test accuracy:  0.9821
    Epoch runtime: 23.78619384765625
    103 Train accuracy: 1.0 Test accuracy:  0.9835
    Epoch runtime: 23.807478666305542
    104 Train accuracy: 1.0 Test accuracy:  0.9824
    Epoch runtime: 23.771175622940063
    105 Train accuracy: 0.9 Test accuracy:  0.9815
    Epoch runtime: 23.747488975524902
    106 Train accuracy: 0.95 Test accuracy:  0.9828
    Epoch runtime: 23.73935627937317
    107 Train accuracy: 1.0 Test accuracy:  0.9822
    Epoch runtime: 23.786404132843018
    108 Train accuracy: 0.95 Test accuracy:  0.9823
    Epoch runtime: 23.715999126434326
    109 Train accuracy: 1.0 Test accuracy:  0.9821
    Epoch runtime: 23.805747270584106
    110 Train accuracy: 0.95 Test accuracy:  0.9834
    Epoch runtime: 23.777987957000732
    111 Train accuracy: 1.0 Test accuracy:  0.9832
    Epoch runtime: 23.795848608016968
    112 Train accuracy: 1.0 Test accuracy:  0.9817
    Epoch runtime: 23.806536436080933
    113 Train accuracy: 0.95 Test accuracy:  0.9836
    Epoch runtime: 23.793235540390015
    114 Train accuracy: 1.0 Test accuracy:  0.9833
    Epoch runtime: 23.81822109222412
    115 Train accuracy: 1.0 Test accuracy:  0.9832
    Epoch runtime: 23.88731360435486
    116 Train accuracy: 1.0 Test accuracy:  0.9825
    Epoch runtime: 23.792627811431885
    117 Train accuracy: 0.9 Test accuracy:  0.9834
    Epoch runtime: 24.039629459381104
    118 Train accuracy: 1.0 Test accuracy:  0.9821
    Epoch runtime: 23.766864776611328
    119 Train accuracy: 1.0 Test accuracy:  0.983
    Epoch runtime: 23.819640398025513
    120 Train accuracy: 0.95 Test accuracy:  0.9824
    Epoch runtime: 23.779805183410645
    121 Train accuracy: 1.0 Test accuracy:  0.9833
    Epoch runtime: 23.78531002998352
    122 Train accuracy: 1.0 Test accuracy:  0.9824
    Epoch runtime: 23.805232763290405
    123 Train accuracy: 0.95 Test accuracy:  0.9833
    Epoch runtime: 23.80686616897583
    124 Train accuracy: 0.95 Test accuracy:  0.9833
    Epoch runtime: 23.810511589050293
    125 Train accuracy: 1.0 Test accuracy:  0.9831
    Epoch runtime: 23.77710509300232
    126 Train accuracy: 1.0 Test accuracy:  0.9824
    Epoch runtime: 23.82640314102173
    127 Train accuracy: 0.95 Test accuracy:  0.9831
    Epoch runtime: 23.82068181037903
    128 Train accuracy: 0.95 Test accuracy:  0.9823
    Epoch runtime: 23.808733224868774
    129 Train accuracy: 1.0 Test accuracy:  0.9827
    Epoch runtime: 23.827823162078857
    130 Train accuracy: 1.0 Test accuracy:  0.9843
    Epoch runtime: 23.804529666900635
    131 Train accuracy: 1.0 Test accuracy:  0.9836
    Epoch runtime: 23.784154891967773
    132 Train accuracy: 1.0 Test accuracy:  0.9836
    Epoch runtime: 23.84726357460022
    133 Train accuracy: 1.0 Test accuracy:  0.984
    Epoch runtime: 23.925867319107056
    134 Train accuracy: 1.0 Test accuracy:  0.9828
    Epoch runtime: 23.847952604293823
    135 Train accuracy: 1.0 Test accuracy:  0.9832
    Epoch runtime: 23.796183109283447
    136 Train accuracy: 1.0 Test accuracy:  0.9827
    Epoch runtime: 23.802472829818726
    137 Train accuracy: 1.0 Test accuracy:  0.9838
    Epoch runtime: 23.77868676185608
    138 Train accuracy: 1.0 Test accuracy:  0.9836
    Epoch runtime: 23.813351154327393
    139 Train accuracy: 0.95 Test accuracy:  0.9811
    Epoch runtime: 23.819189071655273
    140 Train accuracy: 1.0 Test accuracy:  0.9838
    Epoch runtime: 23.79629397392273
    141 Train accuracy: 1.0 Test accuracy:  0.9831
    Epoch runtime: 23.804064989089966
    142 Train accuracy: 1.0 Test accuracy:  0.9847
    Epoch runtime: 23.802648067474365
    143 Train accuracy: 1.0 Test accuracy:  0.9843
    Epoch runtime: 23.968812942504883
    144 Train accuracy: 1.0 Test accuracy:  0.9839
    Epoch runtime: 23.774815797805786
    145 Train accuracy: 1.0 Test accuracy:  0.9842
    Epoch runtime: 23.807541131973267
    146 Train accuracy: 1.0 Test accuracy:  0.9827
    Epoch runtime: 23.76044726371765
    147 Train accuracy: 0.95 Test accuracy:  0.9836
    Epoch runtime: 23.786118745803833
    148 Train accuracy: 1.0 Test accuracy:  0.982
    Epoch runtime: 23.812981843948364
    149 Train accuracy: 1.0 Test accuracy:  0.9841
    Epoch runtime: 23.797524213790894
    Best Epoch: 142
    Best Test: 0.9847


### Dropout

Dropout works by, during training, randomly excluding some of the neurons from each of the input and internal layers (but not the output).  This has the effect of forcing the model to train on many different neural network configurations with many different augmented instances of the training data.  This is somewhat similar to bagging, where many different models are all trained on the same data and, then the prediction is based on agreement between these models.  The models will be forced to not rely excessivly on any one neuron weight or any one feature of the input data.  

The number of neurons that are excluded is based on a hyper-parameter $p$, the probability of a neuron being excluded.  $p = 0.50$ is a typical setting point.

During test (or inference) none of the neurons or inputs are excluded, this increases the overall input value to the next layer.  To compensate for this, during inference, the inputs must be multiplied by the *keep probability* $(1-p)$.



#### Model Setup: Dropout


```python
#Always reset graph (unless managing multiple graphs)
reset_graph()

from tensorflow.contrib.layers import batch_norm
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

#setup naming convention for TensorBoard logs
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/ANN_Basics_run-{}/".format(root_logdir, now)

he_init = tf.contrib.layers.variance_scaling_initializer()

#setup Neural Net Layer function with Tensorboard hooks
def fc_layer(input, channels_in, channels_out, name):
    with tf.name_scope(name):
        w = tf.Variable(he_init([channels_in, channels_out]), name="W") #random values to initialize weights
        b = tf.Variable(tf.zeros([channels_out]), name="B")
        U = tf.matmul(input,w)+b
        z = batch_norm(U, is_training=is_training, decay=0.9, updates_collections=None, scale=True)
        act = tf.nn.elu(z)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act
    
def logit_layer(input, channels_in, channels_out, name):
    with tf.name_scope(name):
        w = tf.Variable(he_init([channels_in, channels_out]), name="W") #random values to initialize weights
        b = tf.Variable(tf.zeros([channels_out]), name="B")
        U = tf.matmul(input,w) + b
        act = batch_norm(U, is_training=is_training, decay=0.9, updates_collections=None)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

#Setup input variable definitions
#MNIST numbers example
height = 28
width = 28
channels = 1
n_inputs = height * width
```

#### Layer Definition: Dropout


```python
from tensorflow.contrib.layers import dropout

#define layer sizes
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
keep_prob = 0.5

#define placeholders for the training data set
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

#define model layers
with tf.name_scope("DNN"):                        #name_scope helps orgainize and clarify TensorBoard graph
    X_drop = dropout(X, keep_prob=keep_prob, is_training=is_training)
    hidden1 = fc_layer(X_drop, n_inputs, n_hidden1, name="hidden1")
    hidden1_drop = dropout(hidden1, keep_prob=keep_prob, is_training = is_training)
    hidden2 = fc_layer(hidden1, n_hidden1, n_hidden2, name="hidden2")
    hidden2_drop = dropout(hidden2, keep_prob=keep_prob, is_training=is_training)
    logits = logit_layer(hidden2_drop, n_hidden2, n_outputs, name="logits")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")
```

#### Loss & Training Functions: Batch Normalization


```python
#define loss function
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
#tf.summary.scalar('cross_entropy', xentropy)

#define training operation
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

#define measure of accuracy to track
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
accuracy_summary = tf.summary.scalar('accuracy', accuracy)
```

#### Model Log: Batch Normalization


```python
#Create initialization function
init = tf.global_variables_initializer()

#Initialize model saver
saver = tf.train.Saver()

#Initialize TensorBoard file writer and collect all data hooks
X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
#tf.summary.image('input', X_reshaped, 3)
merged_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(logdir + 'train/', tf.get_default_graph())
test_writer = tf.summary.FileWriter(logdir + 'test/')
```

#### Training: Batch Normalization


```python
import time
import input_data
mnist = input_data.read_data_sets("MNIST_data/")

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

n_epochs = 20
batch_size = 20

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        start = time.time()
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run([training_op, extra_update_ops], feed_dict={is_training: True, X: X_batch, y:y_batch})
            if iteration % 10 == 0:
                summary_str = sess.run(merged_summary, feed_dict={is_training: True, X: X_batch, 
                                                          y: y_batch})
                step = epoch * mnist.train.num_examples // batch_size + iteration
                train_writer.add_summary(summary_str, step)
                summary_str, acc = sess.run([accuracy_summary, accuracy], feed_dict={is_training: False,
                                                                                     X: mnist.test.images, 
                                                                                     y:mnist.test.labels})
                test_writer.add_summary(summary_str, step)
        acc_train = accuracy.eval(feed_dict={is_training: True, X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={is_training: False, X: mnist.test.images, y:mnist.test.labels})
        
        
        print(epoch, "Train accuracy:" , acc_train, "Test accuracy: ", acc_test)
        end = time.time()
        print("Epoch runtime:", (end-start))
        
    save_path = saver.save(sess, "./my_model_final.ckpt")
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
    0 Train accuracy: 0.75 Test accuracy:  0.8794
    Epoch runtime: 24.720505714416504
    1 Train accuracy: 0.8 Test accuracy:  0.8898
    Epoch runtime: 24.428821086883545
    2 Train accuracy: 0.8 Test accuracy:  0.8996
    Epoch runtime: 24.474013805389404
    3 Train accuracy: 0.6 Test accuracy:  0.9018
    Epoch runtime: 24.401673078536987
    4 Train accuracy: 0.7 Test accuracy:  0.9014
    Epoch runtime: 24.447490692138672
    5 Train accuracy: 0.7 Test accuracy:  0.9037
    Epoch runtime: 24.379454612731934
    6 Train accuracy: 0.8 Test accuracy:  0.9092
    Epoch runtime: 24.410510540008545
    7 Train accuracy: 0.75 Test accuracy:  0.9161
    Epoch runtime: 24.37519598007202
    8 Train accuracy: 0.8 Test accuracy:  0.9083
    Epoch runtime: 24.415316581726074
    9 Train accuracy: 0.85 Test accuracy:  0.9163
    Epoch runtime: 24.441454887390137
    10 Train accuracy: 0.75 Test accuracy:  0.9248
    Epoch runtime: 24.40226435661316
    11 Train accuracy: 0.9 Test accuracy:  0.9178
    Epoch runtime: 24.436115026474
    12 Train accuracy: 0.95 Test accuracy:  0.9207
    Epoch runtime: 24.403642892837524
    13 Train accuracy: 0.8 Test accuracy:  0.9239
    Epoch runtime: 24.42358160018921
    14 Train accuracy: 0.85 Test accuracy:  0.9272
    Epoch runtime: 24.423075914382935
    15 Train accuracy: 0.85 Test accuracy:  0.931
    Epoch runtime: 24.43778920173645
    16 Train accuracy: 0.9 Test accuracy:  0.933
    Epoch runtime: 24.420901775360107
    17 Train accuracy: 0.8 Test accuracy:  0.9295
    Epoch runtime: 24.468000173568726
    18 Train accuracy: 0.85 Test accuracy:  0.9392
    Epoch runtime: 24.66190481185913
    19 Train accuracy: 0.9 Test accuracy:  0.9388
    Epoch runtime: 24.430826425552368


### Data Augmentation

Using the existing training set as inputs, 'new' training samples can be created by augmenting the existing samples.  Samples can be augmented by zooming, shifting, rotating, flipping, adding noise, blurring with a gaussian filter, erroding, adjusting the lighting condition, or applying a perspective transform (affine).

Some of these augmentations may not be appropriate for some applications.  For instance, flipping the MNIST numbers data set will likely cause confusion for some numbers.

### Next Steps

Next major topics to cover will be Dimensionality Reduction, Optimization Algorithms, and specialized networks
