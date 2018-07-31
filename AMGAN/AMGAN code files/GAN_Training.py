# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 16:23:18 2017

@author: Pavitrakumar https://github.com/pavitrakumar78/Anime-Face-GAN-Keras
"""

import os
import glob
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from sklearn.utils import shuffle
import time
import cv2
import scipy
import imageio
from PIL import Image
import matplotlib.gridspec as gridspec
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
from GAN_Nets import get_disc_normal, get_gen_normal #imports architecture of generator and discriminator from the GAN_Nets file
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.backend as K
from scipy.interpolate import spline
K.set_image_dim_ordering('tf')

from collections import deque

np.random.seed(1337)

def norm_img(img):
    img = (img / 127.5) - 1
    return img

def denorm_img(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8)

#GAN does not train on all sample images creates a batch of random images
# data = directory where data is being pulled from data is reference to video game levels taken VGLC, levels have been pre-processed to sub-pieces of 64x64 images
def sample_from_dataset(batch_size, image_shape, data_dir = '/Users/vinay/Anime-Face-GAN-Kerasv1/split_smb_smb2_smb2jp', data = '/Users/vinay/Anime-Face-GAN-Kerasv1/split_smb_smb2_smb2jp/*png'):
    sample_dim = (batch_size,) + image_shape
    sample = np.empty(sample_dim, dtype=np.float32)
    all_data_dirlist = list(glob.glob(data_dir))
    sample_imgs_paths = np.random.choice(all_data_dirlist,batch_size)
    for index,img_filename in enumerate(sample_imgs_paths):
        image = Image.open(img_filename)
        image = image.resize(image_shape[:-1])
        image = image.convert('RGB') #remove transparent ('A') layer
        image = np.asarray(image)
        image = norm_img(image)
        sample[index,...] = image
    return sample


def gen_noise(batch_size, noise_shape):
    return np.random.normal(0, 1, size=(batch_size,)+noise_shape)


def generate_images(generator, save_dir):
    noise = gen_noise(batch_size,noise_shape)
    fake_data_X = generator.predict(noise)
    print("Displaying generated images")
    plt.figure(figsize=(4,4))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(fake_data_X.shape[0],16,replace=False)
    for i in range(16):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = fake_data_X[rand_index, :,:,:]
        fig = plt.imshow(denorm_img(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir+str(time.time())+"_GENERATEDimage.png",bbox_inches='tight',pad_inches=0)


#generated images are plotted in a 4 x 4 grid hence why the output images appear in a grid rather than as seperate images
def save_img_batch(img_batch,img_save_dir):
    plt.figure(figsize=(4,4))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(img_batch.shape[0],16,replace=False)
    for i in range(16):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = img_batch[rand_index, :,:,:]
        fig = plt.imshow(denorm_img(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(img_save_dir,bbox_inches='tight',pad_inches=0)




noise_shape = (1,1,100)
num_steps = 15000 #number of steps the GAN is trained for
batch_size = 64

image_shape = None

img_save_dir = "/Users/vinay/Anime-Face-GAN-Kerasv1/imagesaves" #directory to save generated images

save_model = False




image_shape = (64,64,3)
data_dir =  "/Users/vinay/Anime-Face-GAN-Kerasv1/split_smb_smb2_smb2jp/*png" #directory to pull input images from
log_dir = img_save_dir
save_model_dir = img_save_dir

discriminator = get_disc_normal(image_shape)
generator = get_gen_normal(noise_shape)


discriminator.trainable = False

opt = Adam(lr=0.00015, beta_1=0.5) #same as gen
gen_inp = Input(shape=noise_shape)
GAN_inp = generator(gen_inp)
GAN_opt = discriminator(GAN_inp)
gan = Model(input = gen_inp, output = GAN_opt)
gan.compile(loss = 'binary_crossentropy', optimizer = opt, metrics=['accuracy'])
gan.summary()


avg_disc_fake_loss = deque([0], maxlen=250)
avg_disc_real_loss = deque([0], maxlen=250)
avg_GAN_loss = deque([0], maxlen=250)



for step in range(num_steps):
    tot_step = step
    print("Begin step: ", tot_step)
    step_begin_time = time.time()

    real_data_X = sample_from_dataset(batch_size, image_shape, data_dir = data_dir)

    noise = gen_noise(batch_size,noise_shape)

    fake_data_X = generator.predict(noise)

    if (tot_step % 10) == 0:
        step_num = str(tot_step).zfill(4)
        save_img_batch(fake_data_X,img_save_dir+step_num+"_image.png")


    data_X = np.concatenate([real_data_X,fake_data_X])

    real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2

    fake_data_Y = np.random.random_sample(batch_size)*0.2


    data_Y = np.concatenate((real_data_Y,fake_data_Y))


    discriminator.trainable = True
    generator.trainable = False

    dis_metrics_real = discriminator.train_on_batch(real_data_X,real_data_Y)   #training seperately on real
    dis_metrics_fake = discriminator.train_on_batch(fake_data_X,fake_data_Y)   #training seperately on fake

    print("Disc: real loss: %f fake loss: %f" % (dis_metrics_real[0], dis_metrics_fake[0]))


    avg_disc_fake_loss.append(dis_metrics_fake[0])
    avg_disc_real_loss.append(dis_metrics_real[0])

    generator.trainable = True

    GAN_X = gen_noise(batch_size,noise_shape)

    GAN_Y = real_data_Y

    discriminator.trainable = False

    gan_metrics = gan.train_on_batch(GAN_X,GAN_Y)
    print("GAN loss: %f" % (gan_metrics[0]))

    text_file = open(log_dir+"\\training_log.txt", "a")
    text_file.write("Step: %d Disc: real loss: %f fake loss: %f GAN loss: %f\n" % (tot_step, dis_metrics_real[0], dis_metrics_fake[0],gan_metrics[0]))
    text_file.close()
    avg_GAN_loss.append(gan_metrics[0])
    #records losses to seperate files


    end_time = time.time()
    diff_time = int(end_time - step_begin_time)
    print("Step %d completed. Time took: %s secs." % (tot_step, diff_time))

    if ((tot_step+1) % 500) == 0:
        print("-----------------------------------------------------------------")
        print("Average Disc_fake loss: %f" % (np.mean(avg_disc_fake_loss)))
        print("Average Disc_real loss: %f" % (np.mean(avg_disc_real_loss)))
        print("Average GAN loss: %f" % (np.mean(avg_GAN_loss)))
        print("-----------------------------------------------------------------")
        discriminator.trainable = True
        generator.trainable = True
        generator.save(save_model_dir+str(tot_step)+"_GENERATOR_weights_and_arch.hdf5")
        discriminator.save(save_model_dir+str(tot_step)+"_DISCRIMINATOR_weights_and_arch.hdf5")





#generate final 100 sample images
for i in range(100):
    generate_images(generator, img_save_dir)


"""
#Display Training images sample
save_img_batch(sample_from_dataset(batch_size, image_shape, data_dir = data_dir),img_save_dir+"_12TRAINimage.png")
"""

#Generating GIF from PNG
images = []
all_data_dirlist = list(glob.glob(img_save_dir+"*_image.png"))
for filename in all_data_dirlist:
    img_num = filename.split('\\')[-1][0:-10]
    try:
        if (int(img_num) % 100) == 0:
            images.append(imageio.imread(filename))
    except ValueError:
        pass
imageio.mimsave(img_save_dir+'movie.gif', images)

"""
Alternate way to convert PNG to GIF (ImageMagick):
    >convert -delay 10 -loop 0 *_image.png animated.gif
"""
