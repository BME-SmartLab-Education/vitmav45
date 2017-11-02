'''
Copyright

Jelen forráskód a Budapesti Műszaki és Gazdaságtudományi Egyetemen tartott
"Deep Learning a gyakorlatban Python és LUA alapon" tantárgy segédanyagaként készült.

A tantárgy honlapja: http://smartlab.tmit.bme.hu/oktatas-deep-learning
Deep Learning kutatás: http://smartlab.tmit.bme.hu/deep-learning

A forráskódot GPLv3 licensz védi. Újrafelhasználás esetén lehetőség szerint kérejük
az alábbi szerzőt értesíteni.

2017 (c) Csapó Tamás Gábor (csapot kukac tmit pont bme pont hu)


Original Denoising AutoEncoder example from:

Links:
    [Keras Denoising AutoEncoder] https://blog.keras.io/building-autoencoders-in-keras.html
'''

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model

# ne használja az összes GPU memóriát
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


# FELADAT
# - módosítsd a felhasznált adatbázist: MNIST helyett CIFAR-10
#    (keras.datasets.cifar10)
# - módosítsd a képek méretét a CIFAR-10-hez illeszkedve
# - módosítsd a hálót, hogy RGB csatornás képet tudjon kezelni
#    (az MNIST-ben szürkeárnyalatos képek vannak, a CIFAR-10-ben színesek)
# - keverj hozzá fehérzajt a train és test adatokhoz
#    (Denoising_AE_mnist alapján)
# - ha szükséges, módosítsd a hálót, hogy jobb legyen a zajszűrés
#    (mi segít: több / kevesebb réteg?)

input_img = Input(shape=(28, 28, 1))

# encoder rész, ReLU aktivációs függvénnyel
# itt most több filter van rétegenként a ConvAE-hez képest
x = Convolution2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Convolution2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# modell az encoder-re
encoder = Model(inputs=input_img, outputs=encoded)
# itt (7 x 7 x 32)-es reprezentáció

# decoder rész, ReLU + végén szigmoid aktivációs függvénnyel
x = Convolution2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# modell a teljes AutoEncoder-re (encoder+decoder)
autoencoder = Model(inputs=input_img, outputs=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# train-test adatok

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# FELADAT
x_train_noisy = x_train # + ...
x_test_noisy = x_test # + ...


# tanítás
# bemenet: zajjal kevert képek
# kimenet: eredeti (zajmentes) képek
autoencoder.fit(x_train_noisy, x_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

decoded_imgs = autoencoder.predict(x_test_noisy)

n = 5  # 5 db digit
plt.figure(figsize=(10, 6))
for i in range(n):
    # ábra: eredeti képek
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # ábra: zajos képek
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # ábra: visszaállított képek
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
