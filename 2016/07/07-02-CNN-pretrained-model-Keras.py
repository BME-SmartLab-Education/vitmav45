'''
Copyright

Jelen forráskód a Budapesti Műszaki és Gazdaságtudományi Egyetemen tartott
"Deep Learning a gyakorlatban Python és LUA alapon" tantárgy segédanyagaként készült.

A tantárgy honlapja: http://smartlab.tmit.bme.hu/oktatas-deep-learning
Deep Learning kutatás: http://smartlab.tmit.bme.hu/deep-learning

A forráskódot GPLv3 licensz védi. Újrafelhasználás esetén lehetőség szerint kérejük
az alábbi szerzőt értesíteni.

A kód elkészítéséhez az alábbi források kerültek felhasználásra:
https://keras.io/applications/

2016 (c) Tóth Bálint Pál (toth.b kukac tmit pont bme pont hu)
'''

from urllib.request import urlretrieve
from keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from keras.preprocessing import image
import numpy as np

# kutyás kép letöltése
url_dog="https://pixabay.com/static/uploads/photo/2016/02/19/15/46/dog-1210559_960_720.jpg"
urlretrieve(url_dog, "dog.jpg")
# ImageNet-el előtanított Inception V3 modell betöltése
model = InceptionV3(weights='imagenet', include_top=True)

# kép felkészítése a bemenet számára
img_path = 'dog.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# a kép osztályozása
preds = model.predict(x)
# eredmény kiírása
print('Jósolt osztály:', decode_predictions(preds))
