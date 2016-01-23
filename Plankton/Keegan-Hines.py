__author__ = 'manabchetia'

import os
import sys
import time
import numpy

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import cPickle


from logisticReg import LogisticRegression
from MLP import HiddenLayer, MLP, load_data_simple



class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):


        assert image_shape[1] == filter_shape[1]
        self.input = input


        fan_in = numpy.prod(filter_shape[1:])

        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))


        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )


        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)


        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )


        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )


        relu = lambda x: x * (x > 1e-6)
        self. output=relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))


        self.params = [self.W, self.b]

  def load_data_simple():
    from sklearn.preprocessing import scale
    from sklearn.cross_validation import train_test_split
    from PIL import Image

    with open('./Ocean_50.pkl','rb') as infile:
        full_set =cPickle.load(infile)

    _imgs=full_set[0]
    _imgs=scale(_imgs)
    _labels=full_set[1]



    train_x, test_x, train_y, test_y = train_test_split(_imgs, _labels, test_size=0.2)
    train_set=(train_x,train_y[:,0])
    test_set=(test_x,test_y[:,0])


    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
					      borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
					      borrow=borrow)


        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)

    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y),
            (test_set_x, test_set_y)]
    return rval

  def load_test_data(i):
    from sklearn.preprocessing import scale
    from PIL import Image


    with open('./Ocean_Test_'+str(i)+'.pkl','rb') as infile:
        test_set =cPickle.load(infile)


    _imgs=scale(test_set)




    def shared_dataset(data_x, borrow=True):


        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)


        return shared_x

    test_set_x = shared_dataset(_imgs)


    return test_set_x



  def conv_simple(learning_rate=.003, n_epochs=30,
                           nkerns=[20,50],
                           batch_size=8,with_test_data=True,reg_par=0):

    LEARNING_RATE_SCHEDULE = {
    0: 0.004,
    50000: 0.0004,
    100000: 0.00004,
    }



    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]



    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size


    index = T.lscalar()

    x = T.matrix('x')
    y = T.ivector('y')


       print '... building the model'

    layer0_input = x.reshape((batch_size, 1, 50, 50))
    rng=numpy.random.RandomState()
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 50, 50),
        filter_shape=(nkerns[0], 1, 11, 11),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (nkerns[0], nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 20, 20),
        filter_shape=(nkerns[1], nkerns[0], 13, 13),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=121)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y) + reg_par*(  (layer3.W**2).sum() + (layer2.W**2).sum())


    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )



    params = layer3.params + layer2.params + layer1.params + layer0.params


    grads = T.grad(cost, params)


    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    epoch=0

    while epoch< n_epochs:
        epoch+=1

        for minibatch_index in xrange(n_train_batches):
            it = (epoch - 1) * n_train_batches + minibatch_index
            if it in LEARNING_RATE_SCHEDULE:
                current_LR=LEARNING_RATE_SCHEDULE[it]
                print 'changing learning rate to ' + str(current_LR)
                updates = [
                    (param_i, param_i - current_LR * grad_i)
                    for param_i, grad_i in zip(params, grads)
                ]


            minibatch_avg_cost = train_model(minibatch_index)

             # test it on the test set
        test_losses = [test_model(i) for i in xrange(n_test_batches)]
        print 'test loss: ' + str(numpy.mean(test_losses))
    if with_test_data:
        #free up some memory

        train_set_x.set_value([[]])
        test_set_x.set_value([[]])

        # initialize write file
        with open('Predictions','w') as outfile:
            outfile.write( 'image,' + ','.join(os.listdir('./train/')[1:]) +'\n')

        # grab image file names
        with open('TestData_filename_groupings.pkl','rb') as infile:
            allFileNames=cPickle.load(infile)

        # now loop through the test data files and write predictions
        for i in range(4):
            print 'opening a test file...'
            relevant_file_names=allFileNames[i]
            actual_test_data=load_test_data(i)
            print 'data chunk of size' + str(actual_test_data.get_value(borrow=True).shape)
            n_actual_test_batches = actual_test_data.get_value(borrow=True).shape[0] / batch_size
            print 'n_actual_test_batches = ' + str(n_actual_test_batches)

            print 'computing predictions...'
            predict_model = theano.function(inputs=[index],outputs=layer3.p_y_given_x,givens={x: actual_test_data[index * batch_size: (index + 1) * batch_size]})
            preds=numpy.reshape(numpy.array([predict_model(i) for i in xrange(n_actual_test_batches)]),(batch_size*n_actual_test_batches,121))

            print 'preds.shape = ' + str(preds.shape)

            print 'writing predictions...'
            with open('Predictions','a') as outfile:
                for j,row in enumerate(preds):
                    row=[str(z) for z in row]
                    outfile.write(relevant_file_names[j]+','+','.join(row)+'\n')

            actual_test_data.set_value([[]])



        if __name__ == '__main__':
        	datasets=load_data_simple()
        	conv_simple(reg_par=.001)
