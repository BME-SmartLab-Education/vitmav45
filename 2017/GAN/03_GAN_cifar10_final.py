'''
Copyright

Jelen forráskód a Budapesti Műszaki és Gazdaságtudományi Egyetemen tartott
"Deep Learning a gyakorlatban Python és LUA alapon" tantárgy segédanyagaként készült.

A tantárgy honlapja: http://smartlab.tmit.bme.hu/oktatas-deep-learning
Deep Learning kutatás: http://smartlab.tmit.bme.hu/deep-learning

A forráskódot GPLv3 licensz védi. Újrafelhasználás esetén lehetőség szerint kérejük
az alábbi szerzőt értesíteni.

2017 (c) Csapó Tamás Gábor (csapot kukac tmit pont bme pont hu)

Original example from:
    https://github.com/bstriner/keras-adversarial
'''

import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')

import pandas as pd
import numpy as np
import os
from keras.layers import Reshape, Flatten, LeakyReLU, Activation
from keras.layers.convolutional import UpSampling2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras_adversarial.image_grid_callback import ImageGridCallback

from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from keras_adversarial.legacy import Dense, BatchNormalization, fit, l1l2, Convolution2D, AveragePooling2D
import keras.backend as K
from keras.datasets import cifar10

# ne használja az összes GPU memóriát
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))




def cifar10_process(x):
    x = x.astype(np.float32) / 255.0
    return x


def cifar10_data():
    (xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
    return cifar10_process(xtrain), cifar10_process(xtest)


def dim_ordering_fix(x):
    if K.image_dim_ordering() == 'th':
        return x
    else:
        return np.transpose(x, (0, 2, 3, 1))


def dim_ordering_unfix(x):
    if K.image_dim_ordering() == 'th':
        return x
    else:
        return np.transpose(x, (0, 3, 1, 2))


def dim_ordering_shape(input_shape):
    if K.image_dim_ordering() == 'th':
        return input_shape
    else:
        return (input_shape[1], input_shape[2], input_shape[0])


### GAN hálózat

def model_generator():
    model = Sequential()
    nch = 256
    reg = lambda: l1l2(l1=1e-7, l2=1e-7)
    h = 5
    model.add(Dense(nch * 4 * 4, input_dim=100, W_regularizer=reg()))
    model.add(BatchNormalization(mode=0))
    model.add(Reshape(dim_ordering_shape((nch, 4, 4))))
    model.add(Convolution2D(int(nch / 2), h, h, border_mode='same', W_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(int(nch / 2), h, h, border_mode='same', W_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(int(nch / 4), h, h, border_mode='same', W_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(3, h, h, border_mode='same', W_regularizer=reg()))
    model.add(Activation('sigmoid'))
    return model


def model_discriminator():
    nch = 256
    h = 5
    reg = lambda: l1l2(l1=1e-7, l2=1e-7)

    model = Sequential()
    model.add(Convolution2D(int(nch / 4), h, h, border_mode='same', W_regularizer=reg(),
                       input_shape=dim_ordering_shape((3, 32, 32))))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(int(nch / 2), h, h, border_mode='same', W_regularizer=reg()))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(nch, h, h, border_mode='same', W_regularizer=reg()))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(1, h, h, border_mode='same', W_regularizer=reg()))
    model.add(AveragePooling2D(pool_size=(4, 4), border_mode='valid'))
    model.add(Flatten())
    model.add(Activation('sigmoid'))
    return model


def example_gan(adversarial_optimizer, path, opt_g, opt_d, nb_epoch, generator, discriminator, latent_dim,
                targets=gan_targets, loss='binary_crossentropy'):
    csvpath = os.path.join(path, "history.csv")
    if os.path.exists(csvpath):
        print("Already exists: {}".format(csvpath))
        return

    print("Training: {}".format(csvpath))
    # gan (x - > yfake, yreal), z is gaussian generated on GPU
    # can also experiment with uniform_latent_sampling
    generator.summary()
    discriminator.summary()
    gan = simple_gan(generator=generator,
                     discriminator=discriminator,
                     latent_sampling=normal_latent_sampling((latent_dim,)))

    # build adversarial model
    model = AdversarialModel(base_model=gan,
                             player_params=[generator.trainable_weights, discriminator.trainable_weights],
                             player_names=["generator", "discriminator"])
    model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                              player_optimizers=[opt_g, opt_d],
                              loss=loss)

    # create callback to generate images
    zsamples = np.random.normal(size=(10 * 10, latent_dim))

    def generator_sampler():
        xpred = dim_ordering_unfix(generator.predict(zsamples)).transpose((0, 2, 3, 1))
        return xpred.reshape((10, 10) + xpred.shape[1:])

    generator_cb = ImageGridCallback(os.path.join(path, "epoch-{:03d}.png"), generator_sampler, cmap=None)

    # train model
    xtrain, xtest = cifar10_data()
    y = targets(xtrain.shape[0])
    ytest = targets(xtest.shape[0])
    callbacks = [generator_cb]
    if K.backend() == "tensorflow":
        callbacks.append(
            TensorBoard(log_dir=os.path.join(path, 'logs'), histogram_freq=0, write_graph=True, write_images=True))
    history = fit(model, x=xtrain, y=y, validation_data=(xtest, ytest),
                  callbacks=callbacks, nb_epoch=nb_epoch,
                  batch_size=32)

    # save history to CSV
    df = pd.DataFrame(history.history)
    df.to_csv(csvpath)

    # save models
    generator.save(os.path.join(path, "generator.h5"))
    discriminator.save(os.path.join(path, "discriminator.h5"))


def main():
    # z \in R^100
    latent_dim = 100
    # x \in R^{28x28}
    # generator (z -> x)
    generator = model_generator()
    # discriminator (x -> y)
    discriminator = model_discriminator()
    example_gan(AdversarialOptimizerSimultaneous(), "output/gan-cifar10",
                opt_g=Adam(1e-4, decay=1e-5),
                opt_d=Adam(1e-3, decay=1e-5),
                nb_epoch=100, generator=generator, discriminator=discriminator,
                latent_dim=latent_dim)


if __name__ == "__main__":
    main()
