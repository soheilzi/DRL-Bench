# import jax

import jax.numpy as jnp
import jax
import jaxlib
import numpy as np
import haiku as hk

# import data from ubyte files
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)
    with gzip.open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.frombuffer(lbpath.read(),
                               dtype=np.uint8)

    with gzip.open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.frombuffer(imgpath.read(),
                               dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2

    return images, labels

# load data

train_images, train_labels = load_mnist('data', kind='train')
test_images, test_labels = load_mnist('data', kind='t10k')

#select 2000 samples from training set and 1000 samples from test set
train_images = train_images[0:2000]
train_labels = train_labels[0:2000]

test_images = test_images[0:1000]
test_labels = test_labels[0:1000]

#one-hot encoding
train_labels = np.eye(10)[train_labels]
test_labels = np.eye(10)[test_labels]

# define model
def simple_model(x):
    mlp = hk.Sequential([
        hk.Linear(100),
        jax.nn.relu,
        hk.Linear(10),
        jax.nn.softmax
    ])
    return mlp(x)


# define loss function
# def loss(params, x, y):
#     y_pred = model.apply(params, x)
#     #return jax cross entropy loss function nn
#     return -jnp.mean(jnp.sum(y * jnp.log(y_pred), axis=1))

# # define accuracy function
# def accuracy(params, x, y):
#     y_pred = model.apply(params, x)
#     #return jax cross entropy loss function nn
#     return jnp.mean(jnp.argmax(y_pred, axis=1) == jnp.argmax(y, axis=1))


smodel = hk.without_apply_rng(hk.transform(simple_model))
params = smodel.init(jax.random.PRNGKey(42), jnp.ones((1, 784)))

print("testing values: ", smodel.apply(params, jnp.ones((1, 784))))