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
mpl.use('Agg')

import numpy as np
import os
from keras.layers import Reshape, Flatten, LeakyReLU, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import normal_latent_sampling, AdversarialOptimizerSimultaneous
from keras_adversarial.legacy import l1l2, Dense, fit

from keras.datasets import mnist


# ne használja az összes GPU memóriát
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


def mnist_process(x):
    x = x.astype(np.float32) / 255.0
    return x


def mnist_data():
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    return mnist_process(xtrain), mnist_process(xtest)


# itt kezdődik a GAN


def model_generator(latent_dim, input_shape, hidden_dim=1024, reg=lambda: l1l2(1e-5, 1e-5)):
    return Sequential([
        Dense(int(hidden_dim / 4), input_dim=latent_dim, W_regularizer=reg()),
        LeakyReLU(0.2),
        Dense(int(hidden_dim / 2), W_regularizer=reg()),
        LeakyReLU(0.2),
        Dense(hidden_dim, W_regularizer=reg()),
        LeakyReLU(0.2),
        Dense(np.prod(input_shape), W_regularizer=reg()),
        Activation('sigmoid'),
        Reshape(input_shape)],
        name="generator")


def model_discriminator(input_shape, hidden_dim=1024, reg=lambda: l1l2(1e-5, 1e-5)):
    return Sequential([
        Flatten(input_shape=input_shape),
        Dense(hidden_dim, W_regularizer=reg()),
        LeakyReLU(0.2),
        Dense(int(hidden_dim / 2), W_regularizer=reg()),
        LeakyReLU(0.2),
        Dense(int(hidden_dim / 4), W_regularizer=reg()),
        LeakyReLU(0.2),
        Dense(1, W_regularizer=reg()),
        Activation('sigmoid')],
        name="discriminator")


def example_gan(adversarial_optimizer, path, opt_g, opt_d, nb_epoch, generator, discriminator, latent_dim,
                targets=gan_targets, loss='binary_crossentropy'):


    # gan (x - > yfake, yreal)
    # z generated on GPU
    gan = simple_gan(generator, discriminator, normal_latent_sampling((latent_dim,)))


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
        return generator.predict(zsamples).reshape((10, 10, 28, 28))

    generator_cb = ImageGridCallback(os.path.join(path, "epoch-{:03d}.png"), generator_sampler)

    # train model
    xtrain, xtest = mnist_data()

    # targets = gan_targets -> a 0/1 címkéket rendeli az adatokhoz
    y = targets(xtrain.shape[0])
    ytest = targets(xtest.shape[0])
    callbacks = [generator_cb]
    history = fit(model, x=xtrain, y=y, validation_data=(xtest, ytest), callbacks=callbacks, nb_epoch=nb_epoch,
                  batch_size=32)


# függvények meghívása

# z \in R^100
latent_dim = 100
# x \in R^{28x28}
input_shape = (28, 28)
# generator (z -> x)
generator = model_generator(latent_dim, input_shape)
# discriminator (x -> y)
discriminator = model_discriminator(input_shape)
example_gan(AdversarialOptimizerSimultaneous(), "output/gan",
            opt_g=Adam(1e-4, decay=1e-4),
            opt_d=Adam(1e-3, decay=1e-4),
            nb_epoch=100, generator=generator, discriminator=discriminator,
            latent_dim=latent_dim)

