import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torchvision import transforms

seed_val = 1
np.random.seed(seed_val)
torch.manual_seed(seed_val)

root = './data'
if not os.path.exists(root):
    os.makedirs(root)


def get_imgs(n_imgs):
    trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    data = MNIST(root=root, transform=trans, train=True, download=True)
    data_loader = torch.utils.data.DataLoader(
        dataset=data, batch_size=n_imgs, shuffle=True)
    examples = enumerate(data_loader)
    batch_idx, (X_, Y_) = next(examples)
    X = np.squeeze(X_.data.numpy())
    return X.reshape(n_imgs, -1)


def pattern_complete(weights, X, n_iter=10, soft=True):
    Xs = [None] * n_iter
    for i in range(n_iter):
        X = np.dot(X, weights)
        if soft:
            X = np.tanh(X)
        else:
            X[x < 0] = -1
            X[x >= 0] = 1
        Xs[i] = X
    return Xs


def add_noise(x_, noise_level=.2):
    noise_mul = np.random.choice(
        [1, -1], size=len(x_), p=[1-noise_level, noise_level])
    noise_add = np.random.normal(
        size=np.shape(x_), scale=noise_level)
    return x_ * noise_mul + noise_add


'''a demo of the hopfield network'''

# make some patterns
side_len = 28
n_imgs = 2

X = get_imgs(n_imgs)
m, n_units = np.shape(X)

f, axes = plt.subplots(1, n_imgs, figsize=(n_imgs*3, 3))
for i in range(n_imgs):
    axes[i].imshow(X[i].reshape(side_len, side_len), cmap='bone_r')
    axes[i].set_axis_off()
f.suptitle('training patterns')
f.tight_layout()
f.savefig('figs/train.png', dpi=100, bbox_inches='tight')

# memorize the patterns
weights = np.zeros((n_units, n_units))
for x in X:
    weights += np.outer(x, x) / m
weights[np.diag_indices(n_units)] = 0

# make noisy pattern
x_test = X[0]
noise_level = .1
x_test = add_noise(x_test, noise_level=noise_level)

# pattern completion
n_iter = 2
x_hats = pattern_complete(weights, x_test, n_iter=n_iter, soft=True)
x_hats.insert(0, x_test)

f, axes = plt.subplots(1, n_iter+1, figsize=((n_iter+1)*3, 3))
for i in range(n_iter+1):
    axes[i].imshow(
        x_hats[i].reshape(side_len, side_len), cmap='bone_r',
    )
    axes[i].set_axis_off()
    axes[i].set_title(f'pattern completion, iter: {i}')
f.tight_layout()
f.savefig('figs/pc.png', dpi=100, bbox_inches='tight')
