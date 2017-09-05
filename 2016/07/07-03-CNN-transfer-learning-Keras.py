'''
Copyright

Jelen forráskód a Budapesti Műszaki és Gazdaságtudományi Egyetemen tartott
"Deep Learning a gyakorlatban Python és LUA alapon" tantárgy segédanyagaként készült.

A tantárgy honlapja: http://smartlab.tmit.bme.hu/oktatas-deep-learning
Deep Learning kutatás: http://smartlab.tmit.bme.hu/deep-learning

A forráskódot GPLv3 licensz védi. Újrafelhasználás esetén lehetőség szerint kérejük
az alábbi szerzőt értesíteni.

Az Inception V3 modell ebben a cikkben kerül bemutatásra: 
https://arxiv.org/abs/1512.00567

A kód elkészítéséhez az alábbi források kerültek felhasználásra:
https://keras.io/applications/
https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975

Adatokat innen lehet regisztráció után letölteni:
https://www.kaggle.com/c/dogs-vs-cats

Az adatokat az alábbiak szerint kell könyvtárba rendezni:

data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...

2016 (c) Tóth Bálint Pál (toth.b kukac tmit pont bme pont hu)
'''
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import numpy as np

# a bemenő képek mérete (Inception V3 bemenete 299x299)
img_height=299
img_width=299

# a tanító és validációs adatbázis elérési útvonala, teszt adatbázis most nincs
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
# a tanító és validációs adatok száma
nb_train_samples = 1111
nb_validation_samples = 1000
# epoch szám
nb_epoch=50

# előtanított modell betöltése, a fully-connected rétegek nélkül
base_model = InceptionV3(weights='imagenet', include_top=False)
# az utolsó konvolúciós réteg utána egy global average pooling réteget teszünk, ez rögtön "lapítja" (flatten) a 2D konvolúciót
x = base_model.output
x = GlobalAveragePooling2D()(x)
# ezután hozzáadunk egy előrecsatolt réteget ReLU aktivációs függvénnyel
x = Dense(1024, activation='relu')(x)
# és végül egy kimenete lesz a hálónak - a "binary_crossentropy" költségfüggvénynek erre van szüksége
predictions = Dense(1, activation='sigmoid')(x)
# a model létrehozása
model = Model(input=base_model.input, output=predictions)

# két lépésben fogjuk tanítani a hálót
# az első lépésben csak az előrecsatolt rétegeket tanítjuk, a konvolúciós rétegeket befagyasztjuk
for layer in base_model.layers:
    layer.trainable = False
# lefordítjuk a modelt (fontos, hogy ezt a rétegek befagyasztása után csináljuk"
# mivel két osztályunk van, ezért bináris keresztentrópia költségfüggvényt használunk
model.compile(optimizer='adam', loss='binary_crossentropy')

# kép felkészítése a betöltésre és adatdúsításra
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_height, img_width), batch_size=32, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_height, img_width), batch_size=32, class_mode='binary')

# ez a függvény egyszerre végzi az adatdúsítást és a háló tanítását
model.fit_generator(train_generator, samples_per_epoch=nb_train_samples, nb_epoch=nb_epoch, validation_data=validation_generator, nb_val_samples=nb_validation_samples)

# most már van egy célra betanított osztályozónk, ami az Inception V3 előtanított hálót követi
# most jön a második lépés, aminek a során a konvolúciós háló mélyebb rétegeit fagyasztjuk
# felsőbb rétegeit pedig tovább tanítjuk

# ehhez először nézzük meg a háló felépítését
print("Az Inception V3 konvolúciós rétegei:")
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# majd a hálónak csak az első 172 rétegét fagyasztjuk, a többit pedig engedjük tanulni
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# ez után újra le kell fordítanunk a hálót, hogy most már az Inception V3 felsőbb rétegei tanuljanak
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy')

# és ismét indítunk egy tanítást, ezúttal nem csak az előrecsatolt rétegek,
# hanem az Inception V3 felső rétegei is tovább tanulnak
model.fit_generator(train_generator, samples_per_epoch=nb_train_samples, nb_epoch=nb_epoch, validation_data=validation_generator, nb_val_samples=nb_validation_samples)

print("Tanítás vége.")
