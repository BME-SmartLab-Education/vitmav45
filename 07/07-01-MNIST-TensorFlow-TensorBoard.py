'''
Copyright

Jelen forráskód a Budapesti Műszaki és Gazdaságtudományi Egyetemen tartott
"Deep Learning a gyakorlatban Python és LUA alapon" tantárgy segédanyagaként készült.

A tantárgy honlapja: http://smartlab.tmit.bme.hu/oktatas-deep-learning
Deep Learning kutatás: http://smartlab.tmit.bme.hu/deep-learning

A forráskód a hivatalos TensorFlow dokumentáció alapján készült: 
https://www.tensorflow.org/versions/r0.11/how_tos/summaries_and_tensorboard/index.html

A forráskódot GPLv3 licensz védi. Újrafelhasználás esetén lehetőség szerint kérejük
az alábbi szerzőt értesíteni.

2016 (c) Tóth Bálint Pál (toth.b kukac tmit pont bme pont hu)
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

# a log fileoknak a könyvtárak létrehozása, ha még nem léteznek
# ha léteznek, akkor a korábbiakat letörli
summaries_dir='tensorboard'
if tf.gfile.Exists(summaries_dir):
  tf.gfile.DeleteRecursively(summaries_dir)
tf.gfile.MakeDirs(summaries_dir)

sess = tf.Session()
#sess = tf.InteractiveSession()

# bemenet és kimenet definiálása
with tf.name_scope('input'):
  x = tf.placeholder(tf.float32, shape=[None, 784])
  y_ = tf.placeholder(tf.float32, shape=[None, 10])

# súlyok és bias létrehozása és inicializálása
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# konvolúciós és max-pool rétegek létrehozása
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# háló architektúrájának összerakása

# a következő függvénnyel fogjuk a TensorBoard számára 
# rögzíteni a paraméterben átadott változóról az adatokat
def variable_summaries(var, name):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
  with tf.name_scope('stddev'):
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
  tf.scalar_summary('stddev/' + name, stddev)
  tf.scalar_summary('max/' + name, tf.reduce_max(var))
  tf.scalar_summary('min/' + name, tf.reduce_min(var))
  tf.histogram_summary(name, var)

# első konvolúciós réteg 1@28x28,f:5x5@32,z:2x2
with tf.name_scope('input_reshape'):
  x_image = tf.reshape(x, [-1,28,28,1])
  tf.image_summary('input', x_image, 10)

with tf.name_scope('conv1'):
  with tf.name_scope('weights'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    variable_summaries(W_conv1, 'conv1/weights')
  with tf.name_scope('biases'):
    b_conv1 = bias_variable([32])
    variable_summaries(b_conv1, 'conv1/biases')
  with tf.name_scope('Wx_plus_b'):
    preactivate = conv2d(x_image, W_conv1) + b_conv1 # 28x28 ---> 28x28
    tf.histogram_summary('conv1/pre_activations', preactivate)
  with tf.name_scope('ReLU'):
    h_conv1 = tf.nn.relu(preactivate)
    tf.histogram_summary('conv1/activations', h_conv1)
    tf.image_summary('conv1/activation/h_conv1', h_conv1[:,:,:,0:1], 10)
  with tf.name_scope('maxpool'):
    h_pool1 = max_pool_2x2(h_conv1) # 28x28 ---> 14x14
    tf.histogram_summary('conv1/maxpool', h_pool1)

# második konvolúciós réteg 32@14x14,f:5x5@64,z:2x2
with tf.name_scope('conv2'):
  with tf.name_scope('weights'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    variable_summaries(W_conv2, 'conv2/weights')
  with tf.name_scope('biases'):
    b_conv2 = bias_variable([64])
    variable_summaries(b_conv2, 'conv2/biases')
  with tf.name_scope('Wx_plus_b'):
    preactivate = conv2d(h_pool1, W_conv2) + b_conv2 # 14x14 ---> 14x14
    tf.histogram_summary('conv2/pre_activations', preactivate)
  with tf.name_scope('ReLU'):
    h_conv2 = tf.nn.relu(preactivate)
    tf.histogram_summary('conv2/activations', h_conv2)
  with tf.name_scope('maxpool'):
    h_pool2 = max_pool_2x2(h_conv2) # 14x14 ---> 7x7
    tf.histogram_summary('conv2/maxpool', h_pool2)

# fully connected rétegek
with tf.name_scope('fully_connected_1'):
  with tf.name_scope('weights'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    variable_summaries(W_fc1, 'fc1/weights')
  with tf.name_scope('biases'):
    b_fc1 = bias_variable([1024])
    variable_summaries(b_fc1, 'fc1/biases')
  with tf.name_scope('reshape'):
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    tf.histogram_summary('fc1/reshape', h_pool2_flat)
  with tf.name_scope('Wx_plus_b'):
    preactivate = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    tf.histogram_summary('fc1/pre_activations', preactivate)
  with tf.name_scope('ReLU'):
    h_fc1 = tf.nn.relu(preactivate)
    tf.histogram_summary('fc1/activations', h_fc1)

# dropout a kimeneti réteg előtt
with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32)
  tf.scalar_summary('drop_keep_probability',keep_prob)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
                   
# kimeneti réteg
with tf.name_scope('fully_connected_2'):
  with tf.name_scope('weights'):
    W_fc2 = weight_variable([1024, 10])
    variable_summaries(W_fc2, 'fc2/weights')
  with tf.name_scope('biases'):
    b_fc2 = bias_variable([10])
    variable_summaries(b_fc2, 'fc2/biases')
    tf.histogram_summary('fc2/reshape', h_pool2_flat)

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
# Nem használunk softmax+cross_enthropy-t, mert numerikusan instabil. 
# Ehelyett tf.nn.softmax_cross_entropy_with_logits használunk, ami batch-enként átlagolja
# a kimenetet. Ebben a softmax benne van, ezért nem kell külön rétegként definiálni.
# (Softmax+cross entropy esetén így nézne ki:
#    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# )

# tanítás és tesztelés
with tf.name_scope('cross_entropy'):
  diff = tf.nn.softmax_cross_entropy_with_logits(y_conv,y_)
  with tf.name_scope('total'):
    cross_entropy = tf.reduce_mean(diff)
  tf.scalar_summary('cross entropy', cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.scalar_summary('accuracy', accuracy)

# az összes érték összefogása és kiírása a tensorboard könyvtárba
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter(summaries_dir + '/train',sess.graph)
test_writer = tf.train.SummaryWriter(summaries_dir + '/test')

print(sess.run(tf.initialize_all_variables()))
for i in range(1000):
  batch = mnist.train.next_batch(50)
  # minden 10. epochban rögzítjük a summary-kat a teszt adatokon a TensorBoard számára
  if i%10 == 0:
    summary,acc = sess.run([merged, accuracy], feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob: 1.0})
    test_writer.add_summary(summary, i)
    print('Accuracy at epoch %s: %s' % (i, acc))
  else: 
    # rögzítjük az summary-kat a tanító adatokon a TensorBoard számára
    if i % 100 == 99:  # minden 100. epochban rögzítjük a tanítás statisztikáit (ezzel tudjuk majd vizsgálni pl. a terheléseket)
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}, options=run_options, run_metadata=run_metadata)
      train_writer.add_run_metadata(run_metadata, 'epoch%03d' % i)
      train_writer.add_summary(summary, i)
      print('Adding run metadata for epoch', i)
    else: # minden epochban rögzítjük a summary-kat a tanító adatokon a TensorBoardnak
      summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
      train_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()
