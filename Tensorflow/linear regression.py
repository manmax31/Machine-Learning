import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pyplot

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

# Graph Input
X, Y = tf.placeholder("float"), tf.placeholder("float")

# Set Model Weights
W, b = tf.Variable(np.random.randn(), name="weight"), tf.Variable(np.random.randn(), name="bias")

# Construct Model
pred = tf.add(tf.mul(X, W) + b)
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# Gradient Descent
opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Launch the Graph
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for epoch in xrange(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(opt, fee_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step ==0:
            c = sess.run(cost, fee_dict={train_X: x, train_Y: y})
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b)

    print "Optimisation Finished!"
    training_cost = sess.run(cost, fee_dict={X: train_X, Y: train_Y})
    print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'


            
            

