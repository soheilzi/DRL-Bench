# import jax

import jax.numpy as jnp
import jax
import jaxlib
import numpy as np
import haiku as hk
import optax

import numpy as np
import matplotlib.pyplot as plt
import struct
import gzip
import os
import sys
import time
import random
import math
import pickle
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import wandb

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

train_images, train_labels = load_mnist('./data', kind='train')
test_images, test_labels = load_mnist('./data', kind='t10k')

#select 2000 samples from training set and 1000 samples from test set
train_images = train_images
train_labels = train_labels

test_images = test_images
test_labels = test_labels

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


smodel = hk.without_apply_rng(hk.transform(simple_model))
params = smodel.init(jax.random.PRNGKey(42), jnp.ones((1, 784)))

print("testing values: ", smodel.apply(params, jnp.ones((1, 784))))

# define loss function
def loss(params, x, y):
    y_pred = smodel.apply(params, x)
    #return jax cross entropy loss function nn
    return -jnp.mean(jnp.sum(y * jnp.log(y_pred), axis=1))

#define gradient function
grad = jax.grad(loss)

# define optimizer
opt = optax.adam(1e-3)

optstate = opt.init(params)

# #calculate gradient
# grads = jax.grad(loss)(params, train_images, train_labels)

# updates, optstate = opt.update(grads, optstate)

# params = optax.apply_updates(params, updates)

#for loop for training batches
#break data into batches
batch_size = 32
num_batches = train_images.shape[0] // batch_size
num_epochs = 100

#sety up wandb
wandb.init(project="drl-bench", entity="soheilzi")


for epoch in range(num_epochs):
    for j in range(num_batches):
        batch = train_images[j * batch_size:(j + 1) * batch_size]
        batch_labels = train_labels[j * batch_size:(j + 1) * batch_size]
        grads = jax.grad(loss)(params, batch, batch_labels)
        updates, optstate = opt.update(grads, optstate)
        params = optax.apply_updates(params, updates)
    #test accuracy
    y_pred = smodel.apply(params, test_images)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(test_labels, axis=1)
    test_accuracy = np.mean(y_pred == y_true)
    print("Epoch: ", epoch, "Test Accuracy: ", test_accuracy)
    #train accuracy on a random batch
    batch = train_images[0:batch_size]
    batch_labels = train_labels[0:batch_size]
    y_pred = smodel.apply(params, batch)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(batch_labels, axis=1)
    train_accuracy = np.mean(y_pred == y_true)

    print("Epoch: ", epoch, "Train Accuracy: ", train_accuracy)
    #log to wandb
    wandb.log({"test_accuracy": np.float32(test_accuracy), "train_accuracy": np.float32(train_accuracy)})

