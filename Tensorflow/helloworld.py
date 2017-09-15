import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# Constants
hello = tf.constant("Hello World!")
a = tf.constant(2)
b = tf.constant(9)

with tf.Session() as sess:
    print(sess.run(hello))
    print("Addition with constants: {}").format(sess.run(a+b))
    print("Multiplication with constants: {}").format(sess.run(a*b))


# Variables -  Variables need to be defined and then initialized
x = np.random.randint(1000, size=5)
y = tf.Variable(5*x**2 - 3*x + 5, name='y') 

model = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(model)
    print(x)
    print(sess.run(y))


# Placeholders - Values can be passed in later. We feed data into the graph using placeholders
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
mul = tf.mul(a,b)

with tf.Session() as sess:
    print("Addition with Variables: {}").format(sess.run(add, feed_dict={a:2, b:9}))
    print("Multiplication with Variables: {}").format(sess.run(mul, feed_dict={a: 2, b: 9}))
x = tf.placeholder("float", 3) # Store 3 values in the placeholder
y = x * 2

with tf.Session() as session:
    result = session.run(y, feed_dict={x: [1, 2, 3]})
    print(result)


# Arrays
filename = "MarshOrchid.jpg"
raw_image_data = mpimg.imread(filename)

image = tf.placeholder('uint8', [None, None, 1])
slice = tf.slice(image, [1000, 0, 0], [3000, -1, -1])

with tf.Session() as sess:
    result = session.run(slice, feed_dict={image: raw_image_data})
    print(result.shape)

plt.imshow(result)
plt.show()




