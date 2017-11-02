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

from keras.layers import Input, Dense
from keras.models import Model

import numpy as np
import matplotlib.pyplot as plt

import scipy.io.wavfile as io_wav

# ne használja az összes GPU memóriát
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def wavread(filename):
    (Fs, audio) = io_wav.read(filename)
    return (audio, Fs)

def wavwrite(x, Fs, filename):
    scaled = np.int16(x / np.max(np.abs(x)) * 32767)
    io_wav.write(filename, Fs, scaled)

def wavwrite_from_frames(frames, Fs, filename):
    output = np.zeros(len(frames) * len(frames[0]))
    for i in range(len(frames)):
        output[i * len(frames[0]) : (i + 1) * len(frames[0])] += frames[i]
    wavwrite(output, Fs, filename)


# FELADAT
# - készíts egyszerű audio bemenetet
#    pl. 500 másodperc hosszú 1300 Hz-es szinusz jel, 16 kHz-en mintavételezve
# - normalizáld a bemenetet 0 és 1 közé
# - bontsd fel a jelet kisebb szakaszokra (pl. 10 ms)
# - készítsd el a tanító és teszt adatokat megfelelő formátumban
# - próbáld ki, hogy az AutoEncoder hálózat tud-e zajmentesíteni
# - ha a szinuszon sikerült a zajszűrés, próbálkozhatsz bonyolultabb jellel


