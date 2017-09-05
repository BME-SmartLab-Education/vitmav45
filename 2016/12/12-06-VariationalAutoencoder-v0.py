'''
Copyright

Jelen forráskód a Budapesti Műszaki és Gazdaságtudományi Egyetemen tartott
"Deep Learning a gyakorlatban Python és LUA alapon" tantárgy segédanyagaként készült.

A tantárgy honlapja: http://smartlab.tmit.bme.hu/oktatas-deep-learning
Deep Learning kutatás: http://smartlab.tmit.bme.hu/deep-learning

A forráskódot GPLv3 licensz védi. Újrafelhasználás esetén lehetőség szerint kérjük
az alábbi szerzőt értesíteni.

2016 (c) Csapó Tamás Gábor (csapot kukac tmit pont bme pont hu)

Original Variational AutoEncoder example from:
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
    [VAE] https://jmetzen.github.io/2015-11-27/vae.html
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# %matplotlib inline

from VAE import VariationalAutoencoder

np.random.seed(0)
tf.set_random_seed(0)

# MNIST adatok betöltése
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)
n_samples = mnist.train.num_examples


def train(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5):
    
    vae = VariationalAutoencoder(network_architecture, 
                                 learning_rate=learning_rate, 
                                 batch_size=batch_size)
    total_batch = int(n_samples / batch_size)
    
    # tanítási ciklus
    for epoch in range(training_epochs):
        avg_cost = 0.        
        # batch-ek
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)

            # tanítási lépés
            cost = vae.partial_fit(batch_xs)
            
            # hiba számítás
            avg_cost += cost / n_samples * batch_size

        # adott epoch után log kiírása
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), \
                  "cost=", "{:.9f}".format(avg_cost))
    return vae

# TODO: VAE tanítás MNIST adatokon...

