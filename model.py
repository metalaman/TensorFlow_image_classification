import os
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

class image_classifier():

    def __init__(self, images, img_h=128, img_w=128, lr=0.001,
                 batch_size=64, keep_prob=0.75, resume=1, mode='train'):
        '''
        Description: Initializes the model with passes parameters. Loads a
        trained model, if present and resume = 1, to continue training it.
        Parameters: images - list of Training/Val images
                    img_h - height of the images
                    img_w - width of the images
                    lr - learning rate for the optimizer
                    batch_size - specifies the number of images in a batch
                    keep_prob - used to scale the features and turn neurons off
                                in dropout layer
                    resume - if 1 then a pre-trained model is loaded for further
                            training
        Return Value: None
        '''
        self.learning_rate = lr
        self.image_height = img_h
        self.image_width = img_w
        self.batch_size = batch_size
        self.fc_size = 128
        self.num_classes = 150
        self.current_step = 0
        self.current_epoch = 0
        self.num_epoch = 1000
        self.keep_prob = keep_prob
        self.images = images
        self.lb = preprocessing.LabelBinarizer()
        if mode == 'train':
            self.classes = np.array(
                [i.split('.')[0].split('_')[1] for i in images])
            self.classes = self.classes.reshape(-1, 1)
            self.lb.fit(self.classes)
            with open('lb', 'wb') as f:
                pickle.dump(self.lb, f)

            self.inp_dict = {
                "images": tf.placeholder(
                    tf.float32, shape=[None, self.image_height, self.image_width, 3]),
                "classes": tf.placeholder(
                    tf.float32, shape=[None, self.num_classes]),
                "keep_prob": tf.placeholder(
                    tf.float32)
            }
        self.current_epoch = 0
        self.current_step = 0
        self.resume = resume
        if self.resume is 1:
            if os.path.isfile('model/save.npy'):
                self.current_epoch, self.current_step = np.load(
                    "model/save.npy")
            else:
                print "No Checkpoints Available, Restarting Training.."

    def get_batch(self, sess, io):
        '''
        Description: Iterates over the image(Train/Val) list to generate a batch
        and pre-process that batch. Extracts the class of an image from its
        name.
        Parameters: sess - TensorFlow Session object
                    io   - TensorFlow tensor
        Return Value: images_batch - pre-processed batch of training image
                      class_batch  - one-hot encoded true class batch
        '''
        image_path = 'Data/Train/'
        for batch_idx in range(0, len(self.images), self.batch_size):
            images_batch = self.images[batch_idx:batch_idx + self.batch_size]
            class_batch = np.array(
                [i.split('.')[0].split('_')[1] for i in images_batch])
            class_batch = class_batch.reshape(-1, 1)
            images_batch = np.array(map(lambda x: self.load_image(
                sess, io, image_path + x), images_batch))
            class_batch = self.lb.transform(class_batch)
            yield images_batch, class_batch

    def get_test_batch(self, sess, io, test_images):
        '''
        Description: Iterates over the image(Test) list to generate a batch
        and pre-process that batch.
        Parameters: sess - TensorFlow Session object
                    io   - TensorFlow tensor
                    test_images - list of test images
        Return Value: images_batch - pre-processed batch of test images 4-D
        array
        '''
        image_path = 'Data/Test/'
        for batch_idx in range(0, len(test_images), self.batch_size):
            test_batch = test_images[batch_idx:batch_idx +
                                          self.batch_size]
            test_batch = np.array(map(lambda x: self.load_image(
                sess, io, image_path + x), test_batch))
            yield test_batch

    def build_prepro_graph(self):
        '''
        Description: A TensorFlow computational graph. Reads an image file,
        converts it into an 3-D tensor and resizes the image to 128x128 pixels.
        Accepted image formats are JPEG, PNG and GIF.
        Parameters: None
        Return Value: input_file - placeholder containing image's name with path
                      output_jpg - resized JPEG image array.
                      output_png - resized PNG image array.
                      output_gif - resized GIF image array.
        '''
        input_file = tf.placeholder(dtype=tf.string, name="InputFile")
        image_file = tf.read_file(input_file)
        jpg = tf.image.decode_jpeg(image_file, channels=3)
        png = tf.image.decode_png(image_file, channels=3)
        gif = tf.image.decode_gif(image_file)
        output_jpg = tf.image.resize_images(jpg, [128, 128]) / 255.0
        output_jpg = tf.reshape(
            output_jpg, [128, 128, 3], name="Preprocessed_JPG")
        output_png = tf.image.resize_images(png, [128, 128]) / 255.0
        output_png = tf.reshape(
            output_png, [128, 128, 3], name="Preprocessed_PNG")
        output_gif = tf.image.resize_images(gif, [128, 128]) / 255.0
        output_gif = tf.reshape(
            output_gif, [128, 128, 3], name="Preprocessed_GIF")
        return input_file, output_jpg, output_png, output_gif

    def load_image(self, sess, io, image):
        '''
        Description: Computes the build_prepro_graph() for images.
        Parameters: sess - TensorFlow Session object
                    io   - TensorFlow tensor
                    images - name of image with path
        Return Value: returns the array of a pre-processed image.
        '''
        if image.split('.')[-1] == "png":
            return sess.run(io[2], feed_dict={io[0]: image})
        elif image.split('.')[-1] == "gif":
            return sess.run(io[3], feed_dict={io[0]: image})
        return sess.run(io[1], feed_dict={io[0]: image})

    def init_weight(self, shape):
        '''
        Description: Returns a tensor of the given shape, with random initial
        values from a uniform distribution in range [0,1) transformed to a
        range [-1,1).
        Parameters: shape - a list
        Return Value: returns a tensor.
        '''
        return tf.Variable(tf.random_uniform(shape) * 2 - 1)

    def init_bias(self, shape):
        '''
        Description:Creates a tensor of the given shape with all elements set
        to zero.
        Parameters: shape - a list
        Return Value: returns a tensor.
        '''
        return tf.Variable(tf.zeros(shape))

    def conv2D_layer(self, inp, kernel_shape, bias_shape):
        '''
        Description: Computes the 2D convolution on inp and kernel(weights), and
        activates the output with Rectified Linear Unit. Kernel's stride is 1 in
        each dimension of the image and zero padding('SAME') is done to preserve
        the spatial dimension of the input tensor.
        Batch normalization added between the linear and non-linear operations.
        Parameters: inp - a tensor on which 2D convolution is performed.
                    kernel_shape - a list
                    bias_shape - a list
        Return Value: returns a tensor.
        '''
        weights = self.init_weight(kernel_shape)
        bias = self.init_bias(bias_shape)
        conv = tf.nn.conv2d(inp, weights, strides=[1, 1, 1, 1], padding='SAME')
        conv = self.batch_norm_layer(conv, bias_shape)
        return tf.nn.relu(conv + bias)

    def pool_layer(self, inp):
        '''
        Description: Computes the max pool operation on the inp tensor. The
        shape of the pool sliding window is 2x2 to reduce the spatial dimension
        of the inp tensor by half.
        Parameters: inp - a tensor
        Return Value: returns a tensor.
        '''
        return tf.nn.max_pool(inp, ksize=[1, 2, 2, 1], strides=[
                              1, 2, 2, 1], padding='SAME')

    def batch_norm_layer(self, inp, shape):
        '''
        Description: Performs batch normalization on the inp.
        Parameters: inp - a tensor
                    shape - depth of inp
        Return Value: returns a normalized, scaled, offset tensor.
        '''
        mean, var = tf.nn.moments(inp, [0])
        offset = tf.Variable(tf.zeros(shape))
        scale = tf.Variable(tf.ones(shape))
        epsilon = 0.0001
        return tf.nn.batch_normalization(
            inp, mean, var, offset, scale, epsilon)

    def flatten_layer(self, inp):
        '''
        Description: Flattens inp's spatial and depth dimension to a single
        dimension.
        Parameters: inp - a tensor
        Return Value: flat_inp - a tensor with shape [Batch, num_features]
                      num_features - a scalar value which is Height*Width*Depth
        '''
        inp_shape = inp.get_shape()
        num_features = inp_shape[1:4].num_elements()
        flat_inp = tf.reshape(inp, [-1, num_features])
        return flat_inp, num_features

    def fc_layer(self, inp, inp_shape, out_shape, activation=False):
        '''
        Description: Creates a fully connected layer with optional Rectified
        Linear Unit activation for the neurons in the layer.
        Batch normalization added between the linear and non-linear operations.
        Parameters: inp - a tensor
                    inp_shape - shape of inp. 2D, [Batch, num_features]
                    out_shape - determines the number of neurons in the layer
                    activation - Boolean value to perform activation
        Return Value: a tensor
        '''
        weights = self.init_weight([inp_shape, out_shape])
        bias = self.init_bias(out_shape)
        if activation:
            norm = self.batch_norm_layer(tf.matmul(inp, weights), out_shape)
            return tf.nn.relu(norm + bias)
        return tf.matmul(inp, weights) + bias

    def create_feed_dict(self, image_batch, class_batch, keep_prob):
        '''
        Description: Function to create a dict which stores the placeholder
        values for the training/validation graph.
        Parameters: image_batch - a 4D array of batch of images.
                    class_batch - a 2D array of one hot encoded true class
                    keep_prob - used to scale the features and turn neurons off
                                in dropout layer
        Return Value: a dict
        '''
        feed_dict = {}
        feed_dict[self.inp_dict['images']] = image_batch
        feed_dict[self.inp_dict['classes']] = class_batch
        feed_dict[self.inp_dict['keep_prob']] = keep_prob
        return feed_dict

    def build_training_graph(self):
        '''
        Description: A computational graph for training/validation of the model.
        Graph operations have the following flow(in steps with output shape):
        1) Input_Batch      : [batch_size, 128, 128, 3]
        2) conv1            : [batch_size, 128, 128, 32]
        3) conv2            : [batch_size, 128, 128, 32]
        4) pool1            : [batch_size, 64, 64, 32]
        5) conv3            : [batch_size, 64, 64, 32]
        6) conv4            : [batch_size, 64, 64, 32]
        7) pool2            : [batch_size, 32, 32, 32]
        8) conv5            : [batch_size, 32, 32, 64]
        9) conv6            : [batch_size, 32, 32, 64]
        10) pool3           : [batch_size, 16, 16, 64]
        11) flat_pool3      : [batch_size, 16384]
        12) fc1             : [batch_size, 128]
        13) fc2             : [batch_size, 150]
        14) predicted_class : [batch_size, 150]
        cross_entropy loss is evaluated on predicted_class against actual one
        hot encoded classes.
        Parameters: None
        Return Value: loss - cross_entropy loss averaged on all images in the
                             batch.
                      inp_dict - dict which holds the placeholder values for
                                 the current batch

        '''
        conv1 = self.conv2D_layer(self.inp_dict['images'], [3, 3, 3, 32], 32)
        conv2 = self.conv2D_layer(conv1, [3, 3, 32, 32], 32)
        pool1 = self.pool_layer(conv2)

        conv3 = self.conv2D_layer(pool1, [3, 3, 32, 32], 32)
        conv4 = self.conv2D_layer(conv3, [3, 3, 32, 32], 32)
        pool2 = self.pool_layer(conv4)

        conv5 = self.conv2D_layer(pool2, [3, 3, 32, 64], 64)
        conv6 = self.conv2D_layer(conv5, [3, 3, 64, 64], 64)
        pool3 = self.pool_layer(conv6)

        flat_pool3, num_features = self.flatten_layer(pool3)

        fc1 = self.fc_layer(flat_pool3, num_features, self.fc_size, True)
        fc1 = tf.nn.dropout(fc1, self.inp_dict["keep_prob"])
        fc2 = self.fc_layer(fc1, self.fc_size, self.num_classes, False)

        predicted_class = tf.nn.softmax(fc2)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=fc2, labels=self.inp_dict['classes'])
        loss = tf.reduce_mean(cross_entropy)
        #correct_prediction = tf.equal(tf.argmax(predicted_class, dimension=1),
        #    tf.argmax(self.inp_dict['classes'], dimension=1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return loss, self.inp_dict

    def train(self, loss, inp_dict):
        '''
        Description: Function to run the training/validation graph for specified
        epochs. Weight updation is done by Adam Optimizer with a decaying
        learning rate. Also, loads a pre-trained model for resuming training
        and saves the currently trained model after every epoch.
        Parameters: loss - cross_entropy loss averaged on all images in the
                             batch.
                    inp_dict - dict which holds the placeholder values for
                               the current batch
        Return Value: None
        '''
        self.loss = loss
        self.inp_dict = inp_dict
        saver = tf.train.Saver(max_to_keep=10)
        io = self.build_prepro_graph()
        global_step = tf.Variable(
            self.current_step,
            name='global_step',
            trainable=False)
        learning_rate = tf.train.exponential_decay(
            self.learning_rate, global_step, 50000, 0.95, staircase=True)
        optimizer = tf.train.AdamOptimizer(
            learning_rate).minimize(loss, global_step=global_step)
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("learning_rate", learning_rate)
        summary_op = tf.summary.merge_all()

        print 'Begin Training'
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if self.resume is 1:
                print "Loading Previously Trained Model"
                try:
                    ckpt_file = "./model/model.ckpt-" + str(self.current_step)
                    saver.restore(sess, ckpt_file)
                    print "Resuming Training"
                except Exception as e:
                    print str(e).split('\n')[0]
                    print "Checkpoints not found"
                    sys.exit(0)
            writer = tf.summary.FileWriter(
                "model/log_dir/", graph=tf.get_default_graph())

            for epoch in range(self.current_epoch, self.num_epoch):
                loss = []
                batch_iter = self.get_batch(sess, io)
                for batch in xrange(0, len(self.images), self.batch_size):
                    image_batch, actual_class = batch_iter.next()
                    feed_dict = self.create_feed_dict(
                        image_batch, actual_class, self.keep_prob)
                    run = [global_step, optimizer, self.loss, summary_op]
                    step, _, cur_loss, summary = sess.run(
                        run, feed_dict=feed_dict)
                    writer.add_summary(summary, step)
                    if step % 10 == 0:
                        print epoch, ": Global Step:", step, "\tLoss: ", cur_loss
                    loss.append(cur_loss)
                print
                print "Epoch: ", epoch, "\tAverage Loss: ", np.mean(loss)
                print "\nSaving Model..\n"
                saver.save(sess, "./model/model.ckpt", global_step=global_step)
                np.save("model/save", (epoch, step))

    def build_test_graph(self):
        '''
        Description: A computational graph for testing phase of the model.
        Graph operations have the following flow(in steps with output shape):
        1) test_batch       : [batch_size, 128, 128, 3]
        2) conv1            : [batch_size, 128, 128, 32]
        3) conv2            : [batch_size, 128, 128, 32]
        4) pool1            : [batch_size, 64, 64, 32]
        5) conv3            : [batch_size, 64, 64, 32]
        6) conv4            : [batch_size, 64, 64, 32]
        7) pool2            : [batch_size, 32, 32, 32]
        8) conv5            : [batch_size, 32, 32, 64]
        9) conv6            : [batch_size, 32, 32, 64]
        10) pool3           : [batch_size, 16, 16, 64]
        11) flat_pool3      : [batch_size, 16384]
        12) fc1             : [batch_size, 128]
        13) fc2             : [batch_size, 150]
        14) predicted_class : [batch_size, 1]
        Parameters: None
        Return Value: test_batch      - placeholder for test image batch array.
                      predicted_class - a array holding the predicted class of
                                        every image in the test batch.

        '''
        test_batch = tf.placeholder(
            tf.float32, shape=[None, self.image_height, self.image_width, 3])

        conv1 = self.conv2D_layer(test_batch, [3, 3, 3, 32], 32)
        conv2 = self.conv2D_layer(conv1, [3, 3, 32, 32], 32)
        pool1 = self.pool_layer(conv2)

        conv3 = self.conv2D_layer(pool1, [3, 3, 32, 32], 32)
        conv4 = self.conv2D_layer(conv3, [3, 3, 32, 32], 32)
        pool2 = self.pool_layer(conv4)

        conv5 = self.conv2D_layer(pool2, [3, 3, 32, 64], 64)
        conv6 = self.conv2D_layer(conv5, [3, 3, 64, 64], 64)
        pool3 = self.pool_layer(conv6)

        flat_pool3, num_features = self.flatten_layer(pool3)

        fc1 = self.fc_layer(flat_pool3, num_features, self.fc_size, True)
        fc1 = tf.nn.dropout(fc1, 1.0)
        fc2 = self.fc_layer(fc1, self.fc_size, self.num_classes, False)

        predicted_class = tf.nn.softmax(fc2)
        #predicted_class = tf.argmax(tf.nn.softmax(fc2), dimension=1)

        return test_batch, predicted_class

    def test(self, test_batch, predicted_class):
        '''
        Description: Function to run the testing graph. Loads a pre-trained
        model for testing the model. Loads the test images present in Data/Test
        folder. Writes the predicted class for the test images in
        Test_predictions.txt
        Parameters: None
        Return Value: None
        '''
        saver = tf.train.Saver()
        with open('lb') as f:
            self.lb = pickle.load(f)
        if os.path.isfile('model/save.npy'):
            self.current_epoch, self.current_step = np.load(
                "model/save.npy")
        else:
            print "No Checkpoints Available, Train first!"
            sys.exit(0)
        ckpt_file = "./model/model.ckpt-" + str(self.current_step)
        test_images = os.listdir('Data/Test/')
        test_true_classes = [i.split('.')[0].split('_')[1] for i in test_images]
        io = self.build_prepro_graph()
        with open("Test_predictions.txt", 'w') as f:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, ckpt_file)
                batch_iter = self.get_test_batch(sess, io, test_images)
                for batch in xrange(0, len(test_images), self.batch_size):
                    batch_test = batch_iter.next()
                    predicted = sess.run(
                        predicted_class, feed_dict={test_batch: batch_test})
                    predicted = self.lb.inverse_transform(predicted)
                    for i in predicted:
                        f.write(str(i)+'\n')
