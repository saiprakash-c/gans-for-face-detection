import tensorflow as tf

data_dict = np.load('vgg16_weights.npy', encoding='latin1').item()


def discriminator(x):
	with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
		#relu - convolutional layers
		with tf.variable_scope("conv1"):
			conv1_1 = conv_layer(x, "conv1_1")
			conv1_2 = conv_layer(conv1_1, "conv1_2")
			pool1 = max_pool(conv1_2, "pool1")


		with tf.variable_scope("conv2"):
			conv2_1 = conv_layer(pool1, "conv2_1")
			conv2_2 = conv_layer(conv2_1, "conv2_2")
			pool2 = max_pool(conv2_2, "pool2")

		with tf.variable_scope("conv3"):
			conv3_1 = conv_layer(pool2, "conv3_1")
			conv3_2 = conv_layer(conv3_1, "conv3_2")
			conv3_3 = conv_layer(conv3_2, "conv3_3")
			conv3_4 = conv_layer(conv3_3, "conv3_4")
			pool3 = max_pool(conv3_4, "pool3")

		with tf.variable_scope("conv4"):
			conv4_1 = conv_layer(pool3, "conv4_1")
			conv4_2 = conv_layer(conv4_1, "conv4_2")
			conv4_3 = conv_layer(conv4_2, "conv4_3")
			conv4_4 = conv_layer(conv4_3, "conv4_4")
			pool4 = max_pool(conv4_4, "pool4")

		with tf.variable_scope("conv5"):
			conv5_1 = conv_layer(pool4, "conv5_1")
			conv5_2 = conv_layer(conv5_1, "conv5_2")
			conv5_3 = conv_layer(conv5_2, "conv5_3")
			conv5_4 = conv_layer(conv5_3, "conv5_4")
			pool5 = max_pool(conv5_4, "pool5")

		#fully connected layer to determine if the image is fake or real

	    with tf.variable_scope("linear"):
	    	linear = layers.flatten(pool5)
	    	linear = layers.dense(linear, 2, use_bias=False)

	    with tf.variable_scope("out"):
	        out = nn.sigmoid(linear)

	    return out


def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def max_pool(self, bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def conv_layer(self, bottom, name):
    with tf.variable_scope(name):
        filt = self.get_conv_filter(name)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        conv_biases = self.get_bias(name)
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        return relu

"""def fc_layer(self, bottom, name):
    with tf.variable_scope(name):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])

        weights = self.get_fc_weight(name)
        biases = self.get_bias(name)

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return fc"""

def get_conv_filter(self, name):
    return tf.constant(self.data_dict[name][0], name="filter")

def get_bias(self, name):
    return tf.constant(self.data_dict[name][1], name="biases")

"""def get_fc_weight(self, name):
    return tf.constant(self.data_dict[name][0], name="weights")"""


