#!/usr/bin/python3

import time
import random
import os

# supress silly warnings from master branch of tensorflow...
# import sys
# sys.stderr = None

import matplotlib.image as mpimg
import skimage.transform

import torch
import torchvision
from utils import Logger

import tensorflow as tf
from tensorflow import nn, layers
import numpy as np

# Set random seem for reproducibility
manualSeed = 999
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
tf.set_random_seed(manualSeed)

# make dictionary of lists of all bounding boxes
bbox_file = "WIDER_train/wider_face_split/wider_face_train_bbx_gt.txt"

min_image_size = 64
image_size_in = (16, 16, 3)
image_size_up = (64, 64, 3)

#making a dictionary for vgg19 values
data_dict = np.load('vgg19.npy', encoding='latin1').item()

def data_generator():
    with open(bbox_file) as f:
        image_name = None
        bboxes_left = None
        current_image = None
        for line in f:
            line = line.strip()
            if image_name is None:
                image_name = line
                file_name = "WIDER_train/images/{}".format(image_name)
                current_image = mpimg.imread(file_name)
                current_image = np.array(current_image, dtype=np.float32) / 255.0 * 2.0 - 1.0
            elif bboxes_left is None:
                bboxes_left = int(line)
            else:
                numbers = [int(v) for v in line.split(" ")]
                x, y, w, h = numbers[0:4]

                bbox = numbers[0:4]

                # get sub-image and resize
                # ignore faces that are not already smaller than our small size
                if h > min_image_size and w > min_image_size:
                    sub_image = current_image[y:y+h, x:x+w]
                    full_image = skimage.transform.resize(sub_image, (image_size_up[0], image_size_up[1]), anti_aliasing=True, mode="constant")
                    small_image = skimage.transform.resize(sub_image, (image_size_in[0], image_size_in[1]), anti_aliasing=True, mode="constant")
                    full_image = np.rot90(np.array(full_image, dtype=np.float32))
                    small_image = np.rot90(np.array(small_image, dtype=np.float32))

                    yield (full_image, small_image, 1)

                    # then find a random part of the image for a non-face selection
                    total_h = current_image.shape[0]
                    total_w = current_image.shape[1]
                    h = image_size_up[0]
                    w = image_size_up[1]
                    x = random.randint(0, total_w - w)
                    y = random.randint(0, total_h - h)
                    full_image = current_image[y:y+h, x:x+w]
                    small_image = skimage.transform.resize(full_image, (image_size_in[0], image_size_in[1]), anti_aliasing=True, mode="constant")
                    full_image = np.rot90(full_image)
                    small_image = np.rot90(np.array(small_image, dtype=np.float32))

                    yield (full_image, small_image, 0)

                bboxes_left -= 1
                if bboxes_left == 0:
                    image_name = None
                    bboxes_left = None
                    bboxes = []

def batch_generator(batch_size):
    data_gen = data_generator()
    images = []
    small_images = []
    labels = []
    batch_i = 0
    for (image, small_image, label) in data_gen:
        images += [image]
        small_images += [small_image]
        labels += [label]
        if len(images) == batch_size:
            yield (batch_i, (np.stack(images), np.stack(small_images), np.reshape(labels, [-1, 1])))
            images = []
            small_images = []
            labels = []
            batch_i += 1
# os.listdir("somedirectory")

batch_size = 100
num_batches = 39370 / batch_size # approximately

def noise(size):
    return np.random.normal(size=size)

def discriminator(x):
    """Start addition"""
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

        #fully connected layer to determine if the image is fake or real

        with tf.variable_scope("linear"):
            linear = layers.flatten(conv5_4)
            linear = layers.dense(linear, 2, use_bias=False, kernel_initializer=tf.initializers.random_normal(0.0, 0.1))

        with tf.variable_scope("out"):
            out = nn.sigmoid(linear)

        return out


def avg_pool(bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def conv_layer(bottom, name):
    with tf.variable_scope(name):
        filt = get_conv_filter(name)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        conv_biases = get_bias(name)
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        return relu

def get_conv_filter(name):
    return tf.constant(data_dict[name][0], name="filter")

def get_bias(name):
    return tf.constant(data_dict[name][1], name="biases")
"""End of Addition"""

def generator(x):
    with tf.variable_scope("generator_sr", reuse=tf.AUTO_REUSE, initializer=tf.initializers.random_normal(0.0, 0.02)):
        with tf.variable_scope("conv1"):
            conv1 = layers.conv2d(x, 64, 3, strides=1, padding="same")
            conv1 = layers.batch_normalization(conv1, scale=False)
            conv1 = nn.relu(conv1)

        with tf.variable_scope("conv2"):
            # residual block with (up to) 8 convolution layers
            n_conv_layers = 8;

            previous = conv1
            for i in range(n_conv_layers):
                conv2 = layers.conv2d(previous, 64, 3, strides=1, padding="same")
                conv2 = layers.batch_normalization(conv2, scale=False)
                conv2 += previous
                conv2 = nn.relu(conv2)
                previous = conv2

        with tf.variable_scope("conv3"):
            conv3 = layers.conv2d(conv2, 64, 3, strides=1, padding="same")
            conv3 = layers.batch_normalization(conv3, scale=False)
            conv3 = nn.relu(conv3)

        with tf.variable_scope("deconv4"):
            deconv4 = layers.conv2d_transpose(conv3, 256, 3, strides=2, padding="same")
            deconv4 = layers.batch_normalization(deconv4, scale=False)
            deconv4 = nn.relu(deconv4)

        with tf.variable_scope("deconv5"):
            deconv5 = layers.conv2d_transpose(deconv4, 256, 3, strides=2, padding="same")
            deconv5 = layers.batch_normalization(deconv5, scale=False)
            deconv5 = nn.relu(deconv5)

        with tf.variable_scope("conv6"):
            conv6 = layers.conv2d(deconv5, 3, 1, strides=1, padding="same")

        with tf.variable_scope("out"):
            out = nn.tanh(conv6)

    #Adding Refinement network
    with tf.variable_scope("generator_rf", reuse=tf.AUTO_REUSE, initializer=tf.initializers.random_normal(0.0, 0.02)):
        with tf.variable_scope("conv1"):
            conv6 = layers.batch_normalization(conv6, scale=False)
            conv6 = nn.relu(conv6)
            conv1 = layers.conv2d(conv1, 64, 3, strides=1, padding="same")
            conv1 = layers.batch_normalization(conv1, scale=False)
            conv1 = nn.relu(conv1)

        with tf.variable_scope("conv2"):
            # residual block with (up to) 8 convolution layers
            n_conv_layers = 8;

            previous = conv1
            for i in range(n_conv_layers):
                conv2 = layers.conv2d(previous, 64, 3, strides=1, padding="same")
                conv2 = layers.batch_normalization(conv2, scale=False)
                conv2 += previous
                conv2 = nn.relu(conv2)
                previous = conv2

        with tf.variable_scope("conv3"):
            conv3 = layers.conv2d(conv2, 64, 3, strides=1, padding="same")
            conv3 = layers.batch_normalization(conv3, scale=False)
            conv3 = nn.relu(conv3)

        with tf.variable_scope("conv4"):
            conv4 = layers.conv2d(conv3, 256, 3, strides=1, padding="same")
            conv4 = layers.batch_normalization(conv4, scale=False)
            conv4 = nn.relu(conv4)

        with tf.variable_scope("conv5"):
            conv5 = layers.conv2d(conv4, 256, 3, strides=1, padding="same")
            conv5 = layers.batch_normalization(conv5, scale=False)
            conv5 = nn.relu(deconv5)

        with tf.variable_scope("conv6"):
            conv6 = layers.conv2d(conv5, 3, 3, strides=1, padding="same")

        with tf.variable_scope("out"):
            out2 = nn.tanh(conv6)

    return out, out2

# real input (full size)
X = tf.placeholder(tf.float32, shape=(None, ) + image_size_up)
# real labels (face vs non-face)
X_labels = tf.placeholder(tf.float32, shape=(None, 1))
# downsized input image
Z = tf.placeholder(tf.float32, shape=(None, ) + image_size_in)

# Generator
G_sample, G_sample2 = generator(Z)
# Discriminator, has two outputs [face (1.0) vs nonface (0.0), real (1.0) vs generated (0.0)]
D_real = discriminator(X)
D_real_face = tf.slice(D_real, [0, 0], [-1, 1])
D_real_real = tf.slice(D_real, [0, 1], [-1, 1])
D_fake = discriminator(G_sample2)
D_fake_face = tf.slice(D_fake, [0, 0], [-1, 1])
D_fake_real = tf.slice(D_fake, [0, 1], [-1, 1])

# Generator, MSE pixel-wise loss
G_SR_pixel_loss = tf.reduce_mean((G_sample - X)**2)
G_pixel_loss = G_SR_pixel_loss + tf.reduce_mean((G_sample2 - X)**2)
G_adversarial_loss = tf.reduce_mean(
    nn.sigmoid_cross_entropy_with_logits(
        logits=D_fake_real, labels=tf.ones_like(D_fake_real) # * 1.2 - tf.random.uniform(tf.shape(D_fake)) * 0.4
    )
)
G_classification_loss = tf.reduce_mean(
    nn.sigmoid_cross_entropy_with_logits(
        logits=D_fake_face, labels=X_labels
    )
)
G_loss = G_pixel_loss + 0.001 * G_adversarial_loss + 0.01 * G_classification_loss

# Discriminator
D_loss_real = tf.reduce_mean(
    nn.sigmoid_cross_entropy_with_logits(
        logits=D_real_real, labels=tf.ones_like(D_real_real)# * 1.2 - tf.random.uniform(tf.shape(D_real)) * 0.4
    )
)
D_loss_fake = tf.reduce_mean(
    nn.sigmoid_cross_entropy_with_logits(
        logits=D_fake_real, labels=tf.zeros_like(D_fake_real)# + tf.random.uniform(tf.shape(D_fake[:, 0])) * 0.4
    )
)
D_classification_loss_real = tf.reduce_mean(
    nn.sigmoid_cross_entropy_with_logits(
        logits=D_real_face, labels=X_labels
    )
)
D_classification_loss_fake = tf.reduce_mean(
    nn.sigmoid_cross_entropy_with_logits(
        logits=D_fake_face, labels=X_labels
    )
)
D_loss = D_loss_real + D_loss_fake + D_classification_loss_real + D_classification_loss_fake

# Obtain trainable variables for both networks
train_vars = tf.trainable_variables()

G_SR_vars = [var for var in train_vars if 'generator_sr' in var.name]
G_vars = [var for var in train_vars if 'generator' in var.name]
D_vars = [var for var in train_vars if 'discriminator' in var.name]

print("Generator SR parameter count: {}".format(np.sum([np.product(v.get_shape()) for v in G_SR_vars])))
print("Discriminator parameter count: {}".format(np.sum([np.product(v.get_shape()) for v in D_vars])))
print("Generator parameter count: {}".format(np.sum([np.product(v.get_shape()) for v in G_vars])))

learning_rate = tf.placeholder(tf.float32, shape=[])
G_SR_opt = tf.train.AdamOptimizer(learning_rate).minimize(G_SR_pixel_loss, var_list=G_SR_vars)
G_opt = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=G_vars)
D_opt = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=D_vars)

num_test_samples = 25
_, (test_batch, test_small_images, test_labels) = next(batch_generator(num_test_samples))

# Create logger instance
logger = Logger(model_name='FACEGAN')

# Write out ground truth examples, and "dumb" upscaled examples
logger.log_images(
    test_batch, num_test_samples,
    -100, 0, num_batches
)
compare_images = []
for i in range(num_test_samples):
    compare_images += [skimage.transform.resize(test_small_images[i], (image_size_up[0], image_size_up[1]), anti_aliasing=True, mode="constant")]
compare_images = np.stack(compare_images)
logger.log_images(
    compare_images, num_test_samples,
    -100, 1, num_batches
)

# Total number of epochs to train
num_epochs = 10

# Start interactive session
session = tf.InteractiveSession()
# Init Variables
tf.global_variables_initializer().run()

saver = tf.train.Saver()

# Initial SR training by itself
batch_start_time = time.time()
for epoch in range(2):
    lr = 1e-4
    batch_gen = batch_generator(batch_size)
    for n_batch, (real_images, small_images, real_labels) in batch_gen:
        # 2. Train Generator SR
        feed_dict = {X: real_images, Z: small_images, learning_rate: lr}
        _, g_error = session.run([G_SR_opt, G_SR_pixel_loss], feed_dict=feed_dict)

        # Display Progress every few batches
        if n_batch % 2 == 0:
            now_time = time.time()
            elapsed = now_time - batch_start_time
            batch_start_time = now_time
            print("Batches took {:.3f} ms".format(elapsed * 1000))

            test_images = session.run(G_sample, feed_dict={Z: test_small_images})
            test_images = (test_images + 1.0) * 0.5

            logger.log_images(
                test_images, num_test_samples,
                epoch-10, n_batch, num_batches
            )
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                -1, g_error, -1, [-1], [-1]
            )
saver.save(session, "./model_sr_only.ckpt")

real_face_sum = 0
fake_face_sum = 0
total_sum = 0

batch_start_time = time.time()
for epoch in range(num_epochs):
    if epoch > 0:
        saver.save(session, "./model_{}.ckpt".format(epoch))

    lr = 1e-4 if epoch < 5 else 1e-5

    batch_gen = batch_generator(batch_size)
    for n_batch, (real_images, small_images, real_labels) in batch_gen:
        # 1. Train Discriminator
        feed_dict = {X: real_images, X_labels: real_labels, Z: small_images, learning_rate: lr}
        _, d_error, d_pred_real, d_pred_fake, d_real_face, d_fake_face = session.run([D_opt, D_loss, D_real, D_fake, D_real_face, D_fake_face], feed_dict=feed_dict)

        real_face_sum += np.sum(d_real_face.round() == real_labels)
        fake_face_sum += np.sum(d_fake_face.round() == real_labels)
        total_sum += len(real_images)

        # 2. Train Generator
        feed_dict = {X: real_images, X_labels: real_labels, Z: small_images, learning_rate: lr}
        _, g_error = session.run([G_opt, G_loss], feed_dict=feed_dict)

        # Display Progress every few batches
        if n_batch % 2 == 0:
            now_time = time.time()
            elapsed = now_time - batch_start_time
            batch_start_time = now_time
            print("Batches took {:.3f} ms".format(elapsed * 1000))

            test_images = session.run(G_sample2, feed_dict={Z: test_small_images})
            test_images = (test_images + 1.0) * 0.5

            logger.log_images(
                test_images, num_test_samples,
                epoch, n_batch, num_batches
            )
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, 0, d_pred_real, d_pred_fake
            )

            print("Real accuracy ({}/{}) {:.2f}%    Fake accuracy ({}/{}) {:.2f}%".format(real_face_sum, total_sum, real_face_sum / total_sum * 100,
                                                                                          fake_face_sum, total_sum, fake_face_sum / total_sum * 100,))
saver.save(session, "./model_full.ckpt")
