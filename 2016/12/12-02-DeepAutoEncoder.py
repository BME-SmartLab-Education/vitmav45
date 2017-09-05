'''
Copyright

Jelen forráskód a Budapesti Műszaki és Gazdaságtudományi Egyetemen tartott
"Deep Learning a gyakorlatban Python és LUA alapon" tantárgy segédanyagaként készült.

A tantárgy honlapja: http://smartlab.tmit.bme.hu/oktatas-deep-learning
Deep Learning kutatás: http://smartlab.tmit.bme.hu/deep-learning

A forráskódot GPLv3 licensz védi. Újrafelhasználás esetén lehetőség szerint kérejük
az alábbi szerzőt értesíteni.

2016 (c) Csapó Tamás Gábor (csapot kukac tmit pont bme pont hu)


Original AutoEncoder example from:
Using an AutoEncoder on MNIST handwritten digits.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
    [TensorFlow AutoEncoder] https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
'''

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# MNIST adatok betöltése
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

# hálózat tanítási paraméterek
learning_rate = 0.01
training_epochs = 100
batch_size = 256
display_step = 1
examples_to_show = 10

# rejtett rétegeken a neuronok száma
n_hidden = [512, 256, 128, 64]
n_layers_AutoEncoder = len(n_hidden)


n_input = 784 # MNIST adatok mérete (kép: 28*28)

# tf bemenet (csak képek)
X = tf.placeholder("float", [None, n_input])

# súlyok és bias
print('weights')
weights = dict()
weights['encoder_h1'] = tf.Variable(tf.random_normal([n_input, n_hidden[0]]))
print('encoder_h1', n_input, n_hidden[0] )
for i in range(0, n_layers_AutoEncoder - 1):
    weights['encoder_h' + str(i+2)] = tf.Variable(tf.random_normal([n_hidden[i], n_hidden[i + 1]]))
    print('encoder_h' + str(i+2), n_hidden[i], n_hidden[i + 1] )
for i in range(0, n_layers_AutoEncoder - 1):
    weights['decoder_h' + str(i+1)] = tf.Variable(tf.random_normal([n_hidden[n_layers_AutoEncoder - i - 1], n_hidden[n_layers_AutoEncoder - i - 2]]))
    print('decoder_h' + str(i+1), n_hidden[n_layers_AutoEncoder - i - 1], n_hidden[n_layers_AutoEncoder - i - 2])
weights['decoder_h' + str(n_layers_AutoEncoder)] = tf.Variable(tf.random_normal([n_hidden[0], n_input]))
print('decoder_h' + str(n_layers_AutoEncoder), n_hidden[0], n_input)

print('biases')
biases = dict()
for i in range(0, n_layers_AutoEncoder):
    biases['encoder_b' + str(i+1)] = tf.Variable(tf.random_normal([n_hidden[i]]))
    print('encoder_b' + str(i+1), n_hidden[i])
for i in range(0, n_layers_AutoEncoder - 1):
    biases['decoder_b' + str(i+1)] = tf.Variable(tf.random_normal([n_hidden[n_layers_AutoEncoder - i - 2]]))
    print('decoder_b' + str(i+1), n_hidden[n_layers_AutoEncoder - i - 2])
biases['decoder_b' + str(n_layers_AutoEncoder)] = tf.Variable(tf.random_normal([n_input]))
print('decoder_b' + str(n_layers_AutoEncoder), n_input)


# encoder rész, szigmoid aktivációs függvénnyel
def encoder(layer_prev):
    for i in range(0, n_layers_AutoEncoder):
        layer = tf.nn.sigmoid(tf.add(tf.matmul(layer_prev, weights['encoder_h' + str(i+1)]), biases['encoder_b' + str(i+1)]))
        layer_prev = layer
    return layer


# decoder rész, szigmoid aktivációs függvénnyel
def decoder(layer_prev):
    for i in range(0, n_layers_AutoEncoder):
        layer = tf.nn.sigmoid(tf.add(tf.matmul(layer_prev, weights['decoder_h' + str(i+1)]), biases['decoder_b' + str(i+1)]))
        layer_prev = layer
    return layer


# modell összerakása
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# predikció
y_pred = decoder_op
# bemenet visszarakása a kimenetre
y_true = X

# loss, optimizer, MSE
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# változók inicializálása
init = tf.initialize_all_variables()

# ne foglalja le az összes GPU memóriát
config = tf.ConfigProto()
config.gpu_options.allow_growth = True 

with tf.Session(config=config) as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    # tanítási ciklus
    for epoch in range(training_epochs):
        # batch-ek
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # optimalizálás és költség számítás
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # minden epoch után log kiírása
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Encode és decode lépés a teszt halmazon
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Az eredeti és a visszaállított képek összehasonlítása
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.plot()