'''
Copyright

Jelen forráskód a Budapesti Műszaki és Gazdaságtudományi Egyetemen tartott
"Deep Learning a gyakorlatban Python és LUA alapon" tantárgy segédanyagaként készült.

A tantárgy honlapja: http://smartlab.tmit.bme.hu/oktatas-deep-learning
Deep Learning kutatás: http://smartlab.tmit.bme.hu/deep-learning

A forráskódot GPLv3 licensz védi. Újrafelhasználás esetén lehetőség szerint kérejük
az alábbi szerzőt értesíteni.

2017 (c) Csapó Tamás Gábor (csapot kukac tmit pont bme pont hu)

GAN-toy

Original example from:
    http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
    http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html

'''


import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns

sns.set(color_codes=True)

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

### ezek a függvények az animációhoz szükségesek

def samples(
    model,
    session,
    data,
    sample_range,
    batch_size,
    num_points=10000,
    num_bins=100
):
    '''
    Return a tuple (db, pd, pg), where db is the current decision
    boundary, pd is a histogram of samples from the data distribution,
    and pg is a histogram of generated samples.
    '''
    xs = np.linspace(-sample_range, sample_range, num_points)
    bins = np.linspace(-sample_range, sample_range, num_bins)

    # decision boundary
    db = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        db[batch_size * i:batch_size * (i + 1)] = session.run(
            model.D1,
            {
                model.x: np.reshape(
                    xs[batch_size * i:batch_size * (i + 1)],
                    (batch_size, 1)
                )
            }
        )

    # data distribution
    d = data.sample(num_points)
    pd, _ = np.histogram(d, bins=bins, density=True)

    # generated samples
    zs = np.linspace(-sample_range, sample_range, num_points)
    g = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        g[batch_size * i:batch_size * (i + 1)] = session.run(
            model.G,
            {
                model.z: np.reshape(
                    zs[batch_size * i:batch_size * (i + 1)],
                    (batch_size, 1)
                )
            }
        )
    pg, _ = np.histogram(g, bins=bins, density=True)

    return db, pd, pg


def plot_distributions(samps, sample_range):
    db, pd, pg = samps
    # db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))
    f, ax = plt.subplots(1)
    # ax.plot(db_x, db, label='decision boundary')
    ax.set_ylim(0, 1)
    plt.plot(p_x, pd, label='real data')
    plt.plot(p_x, pg, label='generated data')
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()


def save_animation(anim_frames, anim_path, sample_range):
    f, ax = plt.subplots(figsize=(6, 4))
    f.suptitle('1D Generative Adversarial Network', fontsize=15)
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1.4)
    # line_db, = ax.plot([], [], label='decision boundary')
    line_pd, = ax.plot([], [], label='real data')
    line_pg, = ax.plot([], [], label='generated data')
    frame_number = ax.text(
        0.02,
        0.95,
        '',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes
    )
    ax.legend()

    db, pd, _ = anim_frames[0]
    # db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))

    def init():
        # line_db.set_data([], [])
        line_pd.set_data([], [])
        line_pg.set_data([], [])
        frame_number.set_text('')
        return ( line_pd, line_pg, frame_number)

    def animate(i):
        frame_number.set_text(
            'Frame: {}/{}'.format(i, len(anim_frames))
        )
        db, pd, pg = anim_frames[i]
        # line_db.set_data(db_x, db)
        line_pd.set_data(p_x, pd)
        line_pg.set_data(p_x, pg)
        return (line_pd, line_pg, frame_number)

    anim = animation.FuncAnimation(
        f,
        animate,
        init_func=init,
        frames=len(anim_frames),
        blit=True
    )
    anim.save(anim_path, fps=30, extra_args=['-vcodec', 'libx264'])


### ezek a függvények a parancssoros argumentumok kezeléséhez kellenek

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=5000,
                        help='the number of training steps to take')
    parser.add_argument('--hidden-size', type=int, default=4,
                        help='MLP hidden size')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='the batch size')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    parser.add_argument('--anim-path', type=str, default=None,
                        help='path to the output animation file')
    parser.add_argument('--anim-every', type=int, default=1,
                        help='save every Nth frame for animation')
    return parser.parse_args()



### ezek a függvények a háló összerakásához szükségesek

def linear(input, output_dim, scope=None, stddev=1.0):
    with tf.variable_scope(scope):
        w = tf.get_variable(
            'w',
            [input.get_shape()[1], output_dim],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input, w) + b

        
def optimizer(loss, var_list):
    learning_rate = 0.001
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        loss,
        global_step=step,
        var_list=var_list
    )
    return optimizer


def log(x):
    '''
    Sometimes discriminiator outputs can reach values close to
    (or even slightly less than) zero due to numerical rounding.
    This just makes sure that we exclude those values so that we don't
    end up with NaNs during optimisation.
    '''
    return tf.log(tf.maximum(x, 1e-5))




### itt kezdjük a játék GAN összerakását


class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples

# data = DataDistribution()
# plt.hist(data.sample(10000), 100)


# a range a generálás min/max tartománya
class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01

# gen = GeneratorDistribution(N=8)
# plt.hist(gen.sample(10000), 100)




def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
    h1 = linear(h0, 1, 'g1')
    return h1


def discriminator(input, h_dim):
    h0 = tf.nn.relu(linear(input, h_dim * 2, 'd0'))
    h1 = tf.nn.relu(linear(h0, h_dim * 2, 'd1'))
    h2 = tf.nn.relu(linear(h1, h_dim * 2, 'd2'))
    h3 = tf.sigmoid(linear(h2, 1, 'd3'))
    return h3





class GAN(object):
    def __init__(self, params):
        # generátor hálózat:
        # látens z-ből mintavételezés, majd neurális hálónak beadás
        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=(params.batch_size, 1))
            self.G = generator(self.z, params.hidden_size)

        # diszkriminátor hálózat:
        # célja megkülönböztetni a valódi adatokat (self.x)
        # a generált mintáktól (self.z)
        #
        # két másolat kell belőle amiknek közösek a paraméterei,
        # mert TF-ben nem lehet ugyanazt a hálót más bemenettel használni
        self.x = tf.placeholder(tf.float32, shape=(params.batch_size, 1))
        with tf.variable_scope('D'):
            self.D1 = discriminator(self.x, params.hidden_size)
        with tf.variable_scope('D', reuse=True):
            self.D2 = discriminator(self.G, params.hidden_size)

        # loss definiálása a D-re és G-re, majd optimalizálás
        # 2) heurisztikus költség
        self.loss_d = tf.reduce_mean(-log(self.D1) - log(1 - self.D2))
        self.loss_g = tf.reduce_mean(-log(self.D2))

        # TF változókból elkülönítjük a D-hez és G-hez tartozókat
        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]

        self.opt_d = optimizer(self.loss_d, self.d_params)
        self.opt_g = optimizer(self.loss_g, self.g_params)


def train(model, data, gen, params):
    anim_frames = []

    # ne foglalja le az összes GPU memóriát
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        # TF változók inicializálása
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        for step in range(params.num_steps + 1):
            # diszkriminátor frissítése
            x = data.sample(params.batch_size)
            z = gen.sample(params.batch_size)
            loss_d, _, = session.run([model.loss_d, model.opt_d], {
                model.x: np.reshape(x, (params.batch_size, 1)),
                model.z: np.reshape(z, (params.batch_size, 1))
            })

            # generátor frissítése
            z = gen.sample(params.batch_size)
            loss_g, _ = session.run([model.loss_g, model.opt_g], {
                model.z: np.reshape(z, (params.batch_size, 1))
            })


            print('{}: {:.4f}\t{:.4f}'.format(step, loss_d, loss_g))

            # ezek az animációhoz kellenek
            if params.anim_path and (step % params.anim_every == 0):
                anim_frames.append(
                    samples(model, session, data, gen.range, params.batch_size)
                )

        if params.anim_path:
            save_animation(anim_frames, params.anim_path, gen.range)
        else:
            samps = samples(model, session, data, gen.range, params.batch_size)
            plot_distributions(samps, gen.range)



def main(args):
    model = GAN(args)
    train(model, DataDistribution(), GeneratorDistribution(range=8), args)


if __name__ == '__main__':
    main(parse_args())

