"""
discriminator
  picking the grade A data
"""
from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys
import scipy.io as sio
import cv2
# Importing some more libraries
from unity import  colored_hook

def discriminator(input, is_train,reuse=False):
    hidden_units = (64,128,256,512)#(64,128,256,512)#(32,64,128)

    with tf.variable_scope("discrim")as scope:
        if reuse:
            scope.reuse_variables()

        net = input #tf.reshape(input_digit, shape=[-1, pic_size,pic_size, 3])
        with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            normalizer_fn=slim.batch_norm,
            activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
            normalizer_params={"is_training": is_train}):

            for i in range(len(hidden_units)):
                net = slim.conv2d(net, hidden_units[i], [3, 3], scope="conv%d" % (i*2+1))
                net = slim.conv2d(net, hidden_units[i], [3, 3], stride=2, scope="conv%d" % (i*2+2))

            net = slim.flatten(net)
            logits = slim.fully_connected(net, 1, activation_fn=None)
            logits = tf.nn.sigmoid(logits)

    return logits

def main(arv):
    #import dataset
    current_dir = os.getcwd()
    file_dir = os.path.join(current_dir, '32x32/A')
    images = []
    for each in os.listdir(file_dir):
        img = cv2.imread(os.path.join(file_dir,each))
        images.append(np.array(img/255.0))
    A_img = np.array(images)

    file_dir = os.path.join(current_dir, '32x32/F')
    images = []
    for each in os.listdir(file_dir):
        img = cv2.imread(os.path.join(file_dir,each))
        images.append(np.array(img/255.0))
    F_img = np.array(images)

    win_size = 32
    color_ch = 3

    with tf.variable_scope('input'):
        #real and fake image placholders
        good_image = tf.placeholder(tf.float32, shape = [None, 32, 32, 3], name='good_image')
        bad_input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='bad_input')
        is_train = tf.placeholder(tf.bool, name='is_train')

    good_result = discriminator(good_image, is_train)
    bad_result = discriminator(bad_input, is_train, reuse=True)


    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.square(good_result-1) + tf.square(bad_result))
        #loss = -tf.reduce_mean(tf.log(good_result+10e-10)) - tf.reduce_mean(tf.log(1.0-bad_result+10e-10))

    learn_rate = 0.0005   # how fast the model should learn
    slimopt = slim.learning.create_train_op(loss, tf.train.AdamOptimizer(learn_rate))

    # initialising stuff and starting the session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    saver = tf.train.Saver()

    writer = tf.summary.FileWriter("demo/3")
    writer.add_graph(sess.graph)
    summary = tf.summary.merge([
                    tf.summary.scalar("loss_total", loss),
                    tf.summary.scalar("good_result", tf.reduce_mean(good_result)),
                    tf.summary.scalar("bad_result", tf.reduce_mean(bad_result)),
                    tf.summary.histogram("good_hist", good_result),
                    tf.summary.histogram("bad_hist", bad_result),
                    ])

    batch_size = 250  # how many images to use together for training
    hm_epochs =1500  # how many times to go through the entire dataset
    tot_A_imag = 500
    tot_F_imag = 9000
    epoA = int(tot_A_imag/batch_size)
    epoF = int(tot_F_imag/batch_size)

    for epoch in range(hm_epochs):
        epoch_loss = 0    # initializing error as 0
        for i in range(epoF):
            epoch_x = A_img[ (i)%epoA*batch_size : (i%epoA+1)*batch_size ]
            epoch_y = F_img[ i*batch_size : (i+1)*batch_size ]
            _,c = sess.run([slimopt,loss],feed_dict={good_image:epoch_x,
                                            bad_input:epoch_y,
                                            is_train:True})
            epoch_loss += c
        summ = sess.run(summary,feed_dict={good_image:epoch_x,
                                        bad_input:epoch_y,
                                        is_train:False})
        writer.add_summary(summ,epoch)
        if epoch%10==0:
            print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)
        #if i%500 == 0:
        #    if not os.path.exists('./model/' + version):
        #        os.makedirs('./model/' + version)

    print("test-pick")
    file_dir = os.path.join(current_dir, '32x32/data_all')
    images = []
    for each in os.listdir(file_dir):
        img = cv2.imread(os.path.join(file_dir,each))
        images.append(np.array(img/255.0))
    All_img = np.array(images)
    print(All_img.shape[0])
    for i in range(All_img.shape[0]):
        #epoch = All_img[i:i+1]
        #print(epoch.shape)
        result = sess.run(good_result,feed_dict={good_image:All_img[i:i+1],
                                        is_train:False})
        #print(result)
        if result[0]>.575:
            #print("Hey ",result[0])
            cv2.imwrite("32x32/pick/o%d.png"%i, All_img[i]*255.0)


if __name__ == "__main__":
    sys.excepthook = colored_hook(
        os.path.dirname(os.path.realpath(__file__)))
    tf.app.run()
