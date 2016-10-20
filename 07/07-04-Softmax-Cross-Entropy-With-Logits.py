'''
Copyright

Jelen forráskód a Budapesti Műszaki és Gazdaságtudományi Egyetemen tartott
"Deep Learning a gyakorlatban Python és LUA alapon" tantárgy segédanyagaként készült.

A tantárgy honlapja: http://smartlab.tmit.bme.hu/oktatas-deep-learning
Deep Learning kutatás: http://smartlab.tmit.bme.hu/deep-learning

Hivatkozott kód: 07-01-MNIST-TensorFlow-TensorBoard.py 146. sorától

Az alábbi példa egy hallgatói kérdésre válasz és demonstráció,
hogy a softmax+cross-entropy hogyan válik numerikusan instabillá.

Az instabilitást az okozza, hogy a softmax kimenete a bemenő értékek
nagy eltérése esetén 0-ra kerekíti (pl. y_preds első és utolsó elemei esetén),
és a cross-entropy-nál a 0 logaritmusára elszáll. 
Ezt "hackeléssel" ki lehetne küszöbölni, ha figyelnénk a softmax kimenetét
és egy kis értéket hozzáadnánk, ha nullára váltana.
De ennél sokkal elegánsabb a softmax_cross_entropy_with_logits használata,
ami magas precizitás mellett nem engedi a softmax-ot nullára kerekíteni. 

Általánosságban is ezért jobb a softmax + cross-entropy helyett a
softmax_cross_entropy_with_logits-et, vagy ha az összes kimeneti target 
közül tudjuk, hogy mindig csak egy vesz fel 1.0 értéket, az összes többi
0.0-t, akkor a sparse_softmax_cross_entropy_with_logits-et.

A forráskódot GPLv3 licensz védi. Újrafelhasználás esetén lehetőség szerint kérejük
az alábbi szerzőt értesíteni.

2016 (c) Tóth Bálint Pál (toth.b kukac tmit pont bme pont hu)
'''
import tensorflow as tf
import numpy as np

sess = tf.Session()
y_targs = tf.convert_to_tensor(np.array([[0.0, 1.0, 0.0],[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])) # targets
y_preds = tf.convert_to_tensor(np.array([[0.1, 10.0, 0.1],[0.1, 100.0, 0.1],[0.1, 1000.0, 0.1]])) # predictions

y_softmax = tf.nn.softmax(y_preds) # softmax réteg

# softmax + cross-entropy
print("Reduce sum")
loss_separate_1 = -tf.reduce_sum(y_targs * tf.log(y_softmax), reduction_indices=[1])
print(sess.run(loss_separate_1))

print("Reduce mean / reduce sum")
loss_mean_1 = tf.reduce_mean(-tf.reduce_sum(y_targs * tf.log(y_softmax), reduction_indices=[1]))
print(sess.run(loss_mean_1))

# softmax_cross_entropy_with_logits + reduce mean (egy batchen belül)
print("softmax_cross_entropy_with_logits:")
loss_separate_2 = tf.nn.softmax_cross_entropy_with_logits(y_preds, y_targs)
print(sess.run(loss_separate_2))

print("Reduce mean softmax_cross_entropy_with_logits:")
loss_mean_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_preds, y_targs))
print(sess.run(loss_mean_2))


