"""
using stn to improve the result
vae_disentangle using slim on color picture
"""
from __future__ import division
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys
import scipy.io as sio
import cv2
# Importing some more libraries
from transformer import transformer
from unity import  colored_hook,encoder, decoder
import matplotlib.pyplot as plt


    #np.save('num4.npy', xx)
def main(arv):
    def plot_results(model_name="vae_mnist",index = 0):
        import os
        import matplotlib.pyplot as plt
        if not os.path.exists(model_name):
            os.makedirs(model_name)


        filename = os.path.join(model_name, "digits_over_latent.png")
        # display a 30x30 2D manifold of digits
        n = 10
        digit_size = 32
        figure = np.zeros((digit_size * n, digit_size * n,3))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4*4, 4*4, n)
        grid_y = np.linspace(-4*4, 4*4, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.zeros([1,latent_dim])
                z_sample[0,0] = xi
                z_sample[0,1] = yi
                #z_sample = np.array([[xi, yi]])
                x_decoded = sess.run(rec, feed_dict={shf_sample:z_sample, is_train:False})
                #x_decoded = decoder.predict(z_sample)
                digit = x_decoded[0]#.reshape(digit_size, digit_size,3)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size,:] = digit

        plt.figure(figsize=(10, 10))
        start_range = digit_size // 2
        end_range = n * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys')
        plt.savefig(filename)


        filename = os.path.join(model_name, "compair%02d.png"%(index))
        shff = all_shuff[142:260]
        nrml = all_images[142:260]
        num_rows = 5
        num_cols = 3
        num_images = num_rows*num_cols
        plt.figure(figsize=(3*2*num_cols, 2*num_rows))
        for i in range(num_images):
            j = 3*i
            any_image = shff[j]
            any_image0 = nrml[j,:,:,0]
            any_image1 = nrml[j,:,:,1]
            any_image2 = nrml[j,:,:,2]
            x_rolled,x_decoded,x_shuf,error,x_encoded= sess.run([window,rec2,rec,meansq,shf_mean],\
                           feed_dict={input_shuff:[any_image],
                            input_digit0:[any_image0],
                            input_digit1:[any_image1],
                            input_digit2:[any_image2],
                            output_true:[any_image],
                            is_train:False})
            x_tt = nrml[j]#.reshape(pic_size, pic_size)
            x_dec = x_decoded[0]#.reshape(pic_size, pic_size)
            #print(x_encoded.shape)
            sar = [str(int(a*10)/10) for a in x_encoded[0]]
            if len(sar)>5:
                sar = sar[0:5]
            ax = plt.subplot(num_rows, 3*num_cols, 3*i+1)
            plt.imshow(x_rolled[0] ,  cmap='Greys')
            #plt.xlabel(error)
            plt.xlabel('z = ['+", ".join(sar)+']')
            plt.xticks([])
            plt.yticks([])
            ax = plt.subplot(num_rows, 3*num_cols, 3*i+2)
            plt.imshow(x_dec,  cmap='Greys')
            plt.xticks([])
            plt.yticks([])
            ax = plt.subplot(num_rows, 3*num_cols, 3*i+3)
            plt.imshow(x_shuf[0],  cmap='Greys')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig(filename)

        #plt.figure()
        #ax = plt.subplot(1, 2, 1)
        #plt.imshow(x_tt ,  cmap='Greys')
        #plt.xticks([])
        #plt.yticks([])
        #ax = plt.subplot(1, 2, 2)
        #plt.imshow(x_dec,  cmap='Greys')
        #plt.xticks([])
        #plt.yticks([])

        filename = os.path.join(model_name, "style.png")
        plt.figure(figsize=(3*2*num_cols, 2*num_rows))
        glob_a, local_a = sess.run([nrm_sample,shf_sample],\
                           feed_dict={input_shuff:[shff[0]],
                            input_digit0:[nrml[0,:,:,0]],
                            input_digit1:[nrml[0,:,:,1]],
                            input_digit2:[nrml[0,:,:,2]],
                            output_true:[shff[0]],
                            is_train:False})

        for i in range(num_images):
            j = 3*i
            any_image = shff[j]
            any_image0 = nrml[j,:,:,0]
            any_image1 = nrml[j,:,:,1]
            any_image2 = nrml[j,:,:,2]
            origi,hig,wei,rol,xx,yy, glob_b, local_b = sess.run([window,h,w,r,x,y, nrm_sample,shf_sample],\
                           feed_dict={input_shuff:[any_image],
                            input_digit0:[any_image0],
                            input_digit1:[any_image1],
                            input_digit2:[any_image2],
                            output_true:[any_image],
                            is_train:False})

            ab_img = sess.run([rec2],
                           feed_dict={nrm_sample:glob_a,
                            shf_sample:local_b,
                            is_train:False})
            ba_img = sess.run([rec2],
                           feed_dict={nrm_sample:glob_b,
                            shf_sample:local_a,
                            is_train:False})
            ax = plt.subplot(num_rows, 3*num_cols, 3*i+1)
            plt.imshow(origi[0] ,  cmap='Greys')
            plt.xlabel('(%.2f, %.2f) %.2f (%.2f, %.2f))'%(hig,wei,rol,xx,yy))
            plt.xticks([])
            plt.yticks([])
            ax = plt.subplot(num_rows, 3*num_cols, 3*i+2)
            plt.imshow(ab_img[0][0],  cmap='Greys')
            plt.xticks([])
            plt.yticks([])
            ax = plt.subplot(num_rows, 3*num_cols, 3*i+3)
            plt.imshow(ba_img[0][0],  cmap='Greys')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig(filename)

        plt.close('all')
        #plt.show()

    data_type = 4
    if data_type == 3:
        train = sio.loadmat('32x32/train_32x32.mat')
        img_train = np.array(train['X'])
        #y_test = np.array(train['y'])
        color_channel = 3
        tot_images = 60000 # total number of images
    elif data_type == 4:
        current_dir = os.getcwd()
        file_dir = os.path.join(current_dir, '32x32/A')
        images = []
        for each in os.listdir(file_dir):
            img = cv2.imread(os.path.join(file_dir,each))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(np.array(img))
        img_train = np.array(images)#/255
        #print(img_train.shape)
        #win_size = 32
        color_channel = 3
        tot_images = 1500 # total number of images

    #exit()
    # Deciding how many nodes wach layer should have
    pre_transf_units = (8,16,32)
    rec_hidden_units = (32,16)#(512, 256)
    vae_generative_units = (16,32)#(256, 512)
    hidden_units = 64
    latent_dim = 40
    pic_size = 32

    nnn = img_train.shape
    if data_type == 3:
        all_images = np.zeros((nnn[3],pic_size,pic_size,color_channel))
        all_shuff = np.zeros((nnn[3],pic_size,pic_size,color_channel))
        for i in range(nnn[3]):
            all_images[i] = img_train[:,:,:,i].astype('float32')/255.0
            aaa = np.zeros((32,32,3))
            for i1 in range(4):
                for j1 in range(4):
                    px = (3*i1+1)%4
                    py = (3*j1+2)%4
                    aaa[i1*8+0:i1*8+8,j1*8+0:j1*8+8,:] = all_images[i,px*8+0:px*8+8,py*8+0:py*8+8,:]

            all_shuff[i] = aaa
    elif data_type == 4:
        all_images = np.zeros((nnn[0],pic_size,pic_size,color_channel))
        all_shuff = np.zeros((nnn[0],pic_size,pic_size,color_channel))
        for i in range(nnn[0]):
            all_images[i] = img_train[i,:,:,:].astype('float32')/255.0
            aaa = np.zeros((32,32,3))
            for i1 in range(4):
                for j1 in range(4):
                    px = (3*i1+1)%4
                    py = (3*j1+2)%4
                    aaa[i1*8+0:i1*8+8,j1*8+0:j1*8+8,:] = all_images[i,px*8+0:px*8+8,py*8+0:py*8+8,:]

            all_shuff[i] = aaa

    #print("all = ",all_images.shape)
    #print("shuf = ",all_shuff.shape)
    #print(all_images[0])
    #plt.imshow(all_images[1])
    #plt.show()
    #exit()

    with tf.variable_scope("input_layer"):
        input_shuff = tf.placeholder('float32', shape=[None, pic_size,pic_size,color_channel], name = "input_shuff")
        input_digit0 = tf.placeholder('float32', shape=[None, pic_size,pic_size], name = "input_normal0")
        input_digit1 = tf.placeholder('float32', shape=[None, pic_size,pic_size], name = "input_normal1")
        input_digit2 = tf.placeholder('float32', shape=[None, pic_size,pic_size], name = "input_normal2")
        is_train = tf.placeholder(tf.bool, name='is_train')
        print("input",input_digit0.shape)
        input_digit = tf.concat([tf.expand_dims(input_digit0,axis=3),
                                tf.expand_dims(input_digit1,axis=3),
                                tf.expand_dims(input_digit2,axis=3)], axis=3)
        #print("input",input_digit.shape)

    with tf.variable_scope("Pre-transofrm"):
        net = tf.reshape(input_digit, shape=[-1, pic_size,pic_size, 3])
        with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            normalizer_fn=slim.batch_norm,
            activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
            normalizer_params={"is_training": is_train}):

            for i in range(len(pre_transf_units)):
                net = slim.conv2d(net, pre_transf_units[i], [3, 3], scope="conv%d" % (i*2+1))
                net = slim.conv2d(net, pre_transf_units[i], [3, 3], stride=2, scope="conv%d" % (i*2+2))

            net = slim.flatten(net)
            rnn_out = slim.fully_connected(net, 64)

        with tf.variable_scope("scale"):
            with tf.variable_scope("mean"):
                with tf.variable_scope("hidden") as scope:
                    hidden = slim.fully_connected(rnn_out, hidden_units*2, scope=scope)
                with tf.variable_scope("output") as scope:
                    scale_mean = slim.fully_connected(hidden, 2, activation_fn=None, scope=scope)
            scale = tf.nn.sigmoid(scale_mean)*0.25+.85
            h, w = scale[:, 0], scale[:, 1]
            #print(h.shape)

        with tf.variable_scope("shift"):
            with tf.variable_scope("mean"):
                with tf.variable_scope("hidden") as scope:
                    hidden = slim.fully_connected(rnn_out, hidden_units*2, scope=scope)
                with tf.variable_scope("output") as scope:
                    shift_mean = slim.fully_connected(hidden, 2, activation_fn=None, scope=scope)
            shift = tf.nn.tanh(shift_mean)*0.35
            x, y = shift[:, 0], shift[:, 1]

        with tf.variable_scope("roll"):
            with tf.variable_scope("mean"):
                with tf.variable_scope("hidden") as scope:
                    hidden = slim.fully_connected(rnn_out, hidden_units, scope=scope)
                with tf.variable_scope("output") as scope:
                    roll_mean = slim.fully_connected(hidden, 1, activation_fn=None, scope=scope)
            roll = tf.nn.tanh(roll_mean)
            r = roll[:, 0]

    with tf.variable_scope("Transformer"):
        theta = tf.stack([
            tf.concat([tf.stack([h*tf.math.cos(r), tf.math.sin(r)], axis=1), tf.expand_dims(x, 1)], axis=1),
            tf.concat([tf.stack([-tf.math.sin(r), w*tf.math.cos(r)], axis=1), tf.expand_dims(y, 1)], axis=1),
        ], axis=1)

        #print(input_digit.shape)
        #print(tf.reshape(input_digit[:,:,:,0], [-1, pic_size, pic_size,3]).shape)
        window0 = transformer(
            tf.expand_dims(tf.reshape(input_digit0,[-1,pic_size,pic_size]), 3),
            theta, [pic_size, pic_size]
            )[:,:,:,0]
        window1 = transformer(
            tf.expand_dims(tf.reshape(input_digit1,[-1,pic_size,pic_size]), 3),
            theta, [pic_size, pic_size]
            )[:,:,:,0]
        window2 = transformer(
            tf.expand_dims(tf.reshape(input_digit2,[-1,pic_size,pic_size]), 3),
            theta, [pic_size, pic_size]
            )[:,:,:,0]
        window = tf.concat([tf.expand_dims(window0, 3),tf.expand_dims(window1, 3),tf.expand_dims(window2, 3)],axis = 3)
        window = tf.clip_by_value(window,0.0,1.0)

    # VAE model
    with tf.variable_scope("VAE_disent"):
        with tf.variable_scope("shuff"):
            net = input_shuff#tf.reshape(net,[-1,win_size,win_size,1])
            with tf.variable_scope("encoder"):
                shf_mean, shf_log_var = encoder(net,is_train,rec_hidden_units,latent_dim)
            with tf.variable_scope("sampling"):
                shf_standard_sample = tf.random_normal([tf.shape(input_shuff)[0], latent_dim])
                shf_sample = shf_mean + 1*shf_standard_sample * tf.sqrt(tf.exp(shf_log_var))
            with tf.variable_scope("decode"):
                rec  = decoder(shf_sample,is_train, [3,8,8,16],vae_generative_units, latent_dim)

        with tf.variable_scope("normal"):
            net2=window
            with tf.variable_scope("encoder"):
                nrm_mean, nrm_log_var = encoder(net2,is_train,rec_hidden_units,latent_dim*2)
            with tf.variable_scope("sampling"):
                nrm_standard_sample = tf.random_normal([tf.shape(input_digit)[0], latent_dim*2])
                nrm_sample = nrm_mean + 1*nrm_standard_sample * tf.sqrt(tf.exp(nrm_log_var))
                pack_sample = tf.concat([nrm_sample,shf_sample],axis=1)
            with tf.variable_scope("decode"):
                rec2  = decoder(pack_sample,is_train, [3,8,8,16],vae_generative_units, latent_dim*2)
            #rec = tf.reshape(rec,[-1,win_size*win_size*color_channel])

        # output_true shall have the original image for error calculations
        output_true = tf.placeholder('float32', [None, pic_size,pic_size,color_channel], name = "Truth")

    def gaussian_log_likelihood(x, mean, var, eps=1e-8):
        # compute log P(x) for diagonal Guassian
        # -1/2 log( (2pi)^k sig_1 * sig_2 * ... * sig_k ) -  sum_i 1/2sig_i^2 (x_i - m_i)^2
        bb = tf.square(x-mean)
        bb /=(var + eps)
        return -0.5 * tf.reduce_sum( tf.log(2.*np.pi*var + eps)
                                   + bb, axis=1)

    with tf.variable_scope("loss_function"):
        # define our cost function
        with tf.variable_scope("recon_loss"):
            meansq =    tf.reduce_mean(tf.square(rec - output_true))
            meansq *= pic_size*pic_size
            meansq2 =    tf.reduce_mean(tf.square(rec2 - window)*tf.to_float(tf.math.greater(window,1e-3)))
            meansq2 *= pic_size*pic_size
            binarcs = -tf.reduce_mean(
                output_true * tf.log(rec+ 10e-10) +
                (1.0 - output_true) * tf.log(1.0 - rec+ 10e-10))

        with tf.variable_scope("kl_loss"):
            vae_kl = 0.5 * tf.reduce_sum( 0.0 - shf_log_var - 1.0 + tf.exp(shf_log_var)  +
                tf.square(shf_mean - 0.0) , 1)
            vae_kl = tf.reduce_mean(vae_kl)
            vae_kl2 = tf.reduce_mean(
                        - gaussian_log_likelihood(shf_sample, 0.0, 1.0, eps=0.0) \
                        + gaussian_log_likelihood(shf_sample, shf_mean,  (tf.exp(shf_log_var)) )
                        )
            vae_kl3 = 0.5 * tf.reduce_sum( 0.0 - nrm_log_var - 1.0 + tf.exp(nrm_log_var)  +
                tf.square(nrm_mean - 0.0) , 1)
            vae_kl3 = tf.reduce_mean(vae_kl3)
        vae_loss = meansq*.25 + meansq2 + vae_kl + vae_kl3
        #vae_loss = binarcs*0.0001-tf.reduce_mean(tf.square(window))

    learn_rate = 0.001   # how fast the model should learn
    slimopt = slim.learning.create_train_op(vae_loss, tf.train.AdamOptimizer(learn_rate))

    # initialising stuff and starting the session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    saver = tf.train.Saver()


    writer = tf.summary.FileWriter("demo/3")
    writer.add_graph(sess.graph)
    summary = tf.summary.merge([
                    tf.summary.scalar("loss_total", vae_loss),
                    tf.summary.scalar("mean_sq", meansq),
                    tf.summary.scalar("binary_cross", binarcs),
                    tf.summary.image("recon", tf.reshape(rec2, [-1, pic_size, pic_size, 3]) ),
                    tf.summary.image("recon_shuf", tf.reshape(rec, [-1, pic_size, pic_size, 3]) ),
                    tf.summary.image("original", tf.reshape(input_shuff, [-1, pic_size, pic_size, 3]) ),
                    #tf.summary.image("window", tf.reshape(, [-1, win_size, win_size, 1])),
                    ])

    # defining batch size, number of epochs and learning rate
    batch_size = 250  # how many images to use together for training
    hm_epochs = 5000  # how many times to go through the entire dataset
    # running the model for a 1000 epochs taking 100 images in batches
    # total improvement is printed out after each epoch

    kl = 0
    for epoch in range(hm_epochs):
        epoch_loss = 0    # initializing error as 0
        for i in range(int(tot_images/batch_size)):
            epoch_x = all_shuff[ i*batch_size : (i+1)*batch_size ]
            epoch_xx= all_images[ i*batch_size : (i+1)*batch_size ]
            #print("epoch_x",epoch_x.shape)
            epoch_x0= epoch_xx[:,:,:,0]
            epoch_x1= epoch_xx[:,:,:,1]
            epoch_x2= epoch_xx[:,:,:,2]
            _,c = sess.run([slimopt,vae_loss],feed_dict={input_shuff:epoch_x,\
                                            input_digit0:epoch_x0,
                                            input_digit1:epoch_x1,
                                            input_digit2:epoch_x2,
                                            output_true:epoch_x,
                                            is_train:True})
            epoch_loss += c
        epoch_x = all_shuff[110:130]
        epoch_xx= all_images[110:130]
        epoch_x0= epoch_xx[:,:,:,0]
        epoch_x1= epoch_xx[:,:,:,1]
        epoch_x2= epoch_xx[:,:,:,2]
        summ = sess.run(summary, feed_dict={input_shuff: epoch_x, \
                               input_digit0:epoch_x0,
                               input_digit1:epoch_x1,
                               input_digit2:epoch_x2,
                               output_true: epoch_x,
                               is_train:False})
        writer.add_summary(summ,epoch)
        if epoch%100==0:
            print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)
        if epoch%100==0:
            plot_results(model_name="vae_test",index =epoch)
    print('Epoch', epoch+1, '/', hm_epochs, 'loss:',epoch_loss)
    plot_results(model_name="vae_test")

if __name__ == "__main__":
    sys.excepthook = colored_hook(
        os.path.dirname(os.path.realpath(__file__)))
    tf.app.run()
