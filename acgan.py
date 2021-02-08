from __future__ import division, print_function
import time
import matplotlib.pyplot as plt
import numpy as np
import os

from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import Concatenate
from keras.initializers import RandomNormal

import time

(trainX, trainy), (testX, testy) = mnist.load_data()


# define the standalone discriminator model
def define_discriminator(in_shape=(28, 28, 1), n_classes=10):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=in_shape)
    # downsample to 14x14
    fe = Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    # normal
    fe = Conv2D(64, (3, 3), padding='same', kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    # downsample to 7x7
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    # normal
    fe = Conv2D(256, (3, 3), padding='same', kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # real/fake output
    out1 = Dense(1, activation='sigmoid')(fe)
    # class label output
    out2 = Dense(n_classes, activation='softmax')(fe)
    # define model
    model = Model(in_image, [out1, out2])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model


# define the standalone generator model
def define_generator(latent_dim, n_classes=10):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    # linear multiplication
    n_nodes = 7 * 7
    li = Dense(n_nodes, kernel_initializer=init)(li)
    # reshape to additional channel
    li = Reshape((7, 7, 1))(li)
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 7x7 image
    n_nodes = 384 * 7 * 7
    gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
    gen = Activation('relu')(gen)
    gen = Reshape((7, 7, 384))(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li])
    # upsample to 14x14
    gen = Conv2DTranspose(192, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init)(merge)
    gen = BatchNormalization()(gen)
    gen = Activation('relu')(gen)
    # upsample to 28x28
    gen = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init)(gen)
    out_layer = Activation('tanh')(gen)
    # define model
    model = Model([in_lat, in_label], out_layer)
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect the outputs of the generator to the inputs of the discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and label and outputting real/fake and label outputs
    model = Model(g_model.input, gan_output)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model


# load images
def load_real_samples():
    # load dataset
    (trainX, trainy), (_, _) = mnist.load_data()
    # expand to 3d, e.g. add channels
    X = expand_dims(trainX, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    print(X.shape, trainy.shape)
    return [X, trainy]


# select real samples
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))
    return [images, labels_input], y


def save_img(parent_path, epoch, version):
    n_classes = 10
    # generate points in the latent space
    x_input = randn(latent_dim * n_classes)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_classes, latent_dim)
    labels = np.array([i for i in range(0, n_classes)])
    X = generator.predict([z_input, labels])
    X = (X + 1) / 2.0
    r = 2
    c = 5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(X[cnt, :, :, 0], cmap='gray_r')
            axs[i, j].axis('off')
            cnt += 1
    # save plot to file
    filename1 = 'version_' + str(version) + '_epoch_' + str(epoch) + '.png'
    plt.savefig(parent_path + "/" + filename1)
    plt.close()


# generate samples and save as a plot and save the model
D_real_example_binary = []
D_real_example_class = []
D_fake_example_binary = []
D_fake_example_class = []
G_binary = []
G_class = []

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=64):
    sub_images = os.listdir("images")
    new_version = len(sub_images) + 1
    new_parent_path_save_img = "images/version" + str(new_version)
    os.makedirs(new_parent_path_save_img)
    new_path_save_model = "saved_model_weights/version" + str(new_version)
    os.makedirs(new_path_save_model)
    new_path_save_plot_history_training = 'plot_history_training/version' + str(new_version)
    os.makedirs(new_path_save_plot_history_training)

    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)

    # manually enumerate epochs
    for i in range(n_steps):
        # get randomly selected 'real' samples
        [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
        # update discriminator model weights
        _, d_r1, d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
        # generate 'fake' examples
        [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model weights
        _, d_f, d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
        # prepare points in latent space as input for the generator
        [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the generator via the discriminator's error
        _, g_1, g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])

        D_real_example_binary.append(d_r1)
        D_real_example_class.append(d_r2)
        D_fake_example_binary.append(d_f)
        D_fake_example_class.append(d_f2)
        G_binary.append(g_1)
        G_class.append(g_2)

        # summarize loss on this batch
        print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i + 1, d_r1, d_r2, d_f, d_f2, g_1, g_2))
        # evaluate the model performance every 'epoch'
        if i % bat_per_epo == 0:
            current_epoch = i / bat_per_epo
            save_img(new_parent_path_save_img, current_epoch, new_version)

            g_model.save_weights(new_path_save_model + "/generator_weights_" + str(current_epoch) + ".h5")
            d_model.save_weights(new_path_save_model + "/discriminator_weights_" + str(current_epoch) + ".h5")
            gan_model.save_weights(new_path_save_model + "/combined_weights_" + str(current_epoch) + ".h5")
    plot_training_process(new_path_save_plot_history_training)


def plot_training_process(save_path):
    plt.plot(list(range(1, len(D_real_example_binary)+1)), D_real_example_binary, 'b')
    plt.plot(list(range(1, len(D_fake_example_binary)+1)), D_fake_example_binary, 'r')
    plt.title("D_fake_real_loss_on_real_and_fake_image")
    plt.savefig(save_path + '/D_binary_loss.png')
    plt.show()

    plt.plot(list(range(1, len(D_real_example_class)+1)), D_real_example_class, 'b')
    plt.plot(list(range(1, len(D_fake_example_class)+1)), D_fake_example_class, 'r')
    plt.title("D_categorical_loss_on_fake_and_real_image")
    plt.savefig(save_path + '/D_categorical_loss.png')
    plt.show()

    plt.plot(G_binary)
    plt.title("G_fake_real_loss")
    plt.savefig(save_path + "/G_fake_real_loss.png")
    plt.show()

    plt.plot(G_class)
    plt.title("G_categorical_loss")
    plt.savefig(save_path + "/G_categorical_loss.png")
    plt.show()

    from numpy import save
    save(save_path + "/D_real_example_binary.npy", D_real_example_binary)
    save(save_path + "/D_fake_example_binary.npy", D_fake_example_binary)
    save(save_path + "/D_real_example_class.npy", D_real_example_class)
    save(save_path + "/D_fake_example_class.npy", D_fake_example_class)
    save(save_path + "/G_binary.npy", G_binary)
    save(save_path + "/G_class.npy", G_class)


t0 = time.time()
# size of the latent space
latent_dim = 100
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples()
# train model
train(generator, discriminator, gan_model, dataset, latent_dim, n_epochs=100)
t1 = time.time()
print("latency: ", t1 - t0)



