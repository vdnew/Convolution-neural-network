#Convolution Neural Network 
#Date: 3-1-2018
#develop by: vD

import tensorflow as tf 

from tensorflow.examples.tutorials.mnit import input_data

#download and set the variable for the datasets
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


#set parameter
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 5

#parameter for the convolution neural network
n_input = 784
n_classes = 10
dropout = 0.75

#graph parameter
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_preb = tf.placeholder(tf.float32)

def conv2d(x, W, b, strides=1):

	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.n.bias_add(x, b)
	return tf.nn.relu(x)

def maxpool2d(x, k=2):

	return tf.nn.max_pool(x, k=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(x, weights, biases, dropout):

	x = tf.reshpe(x, shape=[-1, 28, 28, 1])

	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	conv1 = maxpool2d(conv1, k=2)

	conv2 = conv2d(x, weights['wc2'], biases['bc2'])
	conv2 = maxpool2d(conv2, k=2)

	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])

	fc1 = tf.nn.relu(fc1)

	fc1 = tf.nn.dropout(fc1, dropout)

	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

	return out


#setting weights biases
weights={
	
	'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
	'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
	'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
	'out': tf.Variable(tf.random_normal([1024, b_classes]))
}

biases={
	
	'bc1': tf.Variable(tf.random_normal([32])),
	'bc2': tf.Variable(tf.random_normal([64])),
	'bd1': tf.Variable(tf.random_normal([1024])),
	'out': tf.Variable(tf.random_normal([n_classes]))

}

pred = conv_net(x, weights, biases, keep_preb)

cost = tf.reduce_mean(tf.nn.softmax_entropy_with_logits(Logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cost(correct_pred, tf.float32))

init = tf.global_variables_initializer()

one = 1
with tf.Session as sess:
	sess.run(init)
	step=1

    while step * batch_size < training_iters:
	batch_x, batch_y = mnist.train.next_batch(batch_size)

	sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

	    if step % display_step == 0:

		loss, acc == sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
	        print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
                        
            step+=one
            print("Optimization finishes!")
            print("Tessting Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))
