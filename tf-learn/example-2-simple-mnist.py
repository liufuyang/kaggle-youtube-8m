# example-2-simple-mnist.py
import tensorflow as tf
from datetime import datetime
import time

# reset everything to rerun in jupyter
tf.reset_default_graph()

# config
batch_size = 100
learning_rate = 0.5
training_epochs = 6
logs_path = './tmp/example-2/' + datetime.now().isoformat()

# load mnist data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('tmp/MNIST_data', one_hot=True)

# input images
with tf.name_scope('input'):
    # None -> batch size can be any size, 784 -> flattened mnist image
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input") 
    # target 10 output classes
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

# model parameters will change during training so we use tf.Variable
with tf.name_scope("weights"):
    W = tf.Variable(tf.zeros([784, 10]))

# bias
with tf.name_scope("biases"):
    b = tf.Variable(tf.zeros([10]))

# implement model
with tf.name_scope("softmax"):
    # y is our prediction
    y = tf.nn.softmax(tf.matmul(x,W) + b)

# specify cost function
with tf.name_scope('cross_entropy'):
    # this is our cost
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# specify optimizer
with tf.name_scope('train'):
    # optimizer is an "operation" which we can execute in a session
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    # Accuracy
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
# create a summary for our cost and accuracy
train_cost_summary = tf.summary.scalar("train_cost", cross_entropy)
train_acc_summary = tf.summary.scalar("train_accuracy", accuracy)
test_cost_summary = tf.summary.scalar("test_cost", cross_entropy)
test_acc_summary = tf.summary.scalar("test_accuracy", accuracy)

# merge all summaries into a single "operation" which we can execute in a session 
# summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    # variables need to be initialized before we can use them
    sess.run(tf.initialize_all_variables())

    # create log writer object
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        
    # perform training cycles
    for epoch in range(training_epochs):
        
        # number of batches in one epoch
        batch_count = int(mnist.train.num_examples/batch_size)
        
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            
            # perform the operations we defined earlier on batch
            _, train_cost, train_acc, _train_cost_summary, _train_acc_summary = 
                sess.run([train_op, cross_entropy, accuracy, train_cost_summary, train_acc_summary], 
                    feed_dict={x: batch_x, y_: batch_y})
            # write log
            writer.add_summary(_train_cost_summary, epoch * batch_count + i)
            writer.add_summary(_train_acc_summary, epoch * batch_count + i)

            if i % 100 == 0:
                # for log on test data:
                test_cost, test_acc, _test_cost_summary, _test_acc_summary = 
                    sess.run([cross_entropy, accuracy, test_cost_summary, test_acc_summary], 
                        feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                # write log
                writer.add_summary(_test_cost_summary, epoch * batch_count + i)
                writer.add_summary(_test_acc_summary, epoch * batch_count + i)
                
                print('Epoch {0:3d}, Batch {1:3d} | Train Cost: {2:.2f} | Test Cost: {3:.2f} | Accuracy batch train: {4:.2f} | Accuracy test: {5:.2f}'
                    .format(epoch, i, train_cost, test_cost, train_acc, test_acc))
            
    print('Accuracy: {}'.format(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})))
    print('done')

# tensooboard --logdir=./tmp/example-2 --port=8002 --reload_interval=5
# You can run the following js code in broswer console to make tensooboard to do auto-refresh
# setInterval(function() {document.getElementById('reload-button').click()}, 5000);
