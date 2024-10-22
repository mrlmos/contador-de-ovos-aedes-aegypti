import time

t1 = time.time()

import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from extrair_imagens import getImagesFromDir
from DensityEstimation import kernel_density_estimation
from pca import PCA


def load_imgs():
    path = "/home/murilo/Documents/Ufs/2024.1/recpad/U1-projeto/dados/base-jhonny/"
    images, labels = getImagesFromDir(path)
    images = images.reshape(images.shape[0], -1)
    return images, labels


def load_test_base():
    path = (
        "/home/murilo/Documents/Ufs/2024.1/recpad/U1-projeto/dados/ovos-mosquito/geral/"
    )
    img, lbl = getImagesFromDir(path)
    img = img.reshape(img.shape[0], -1)
    return img, lbl


imgs, labels = load_imgs()
ic(imgs.shape)
base = np.load("./dados/base_pca.npy")
proj = (imgs @ base).T

x_test, y_test = load_test_base()
x_test = (x_test @ base).T
# ic(np.sum(y_test))

window = 0.8

# Classe 1 = nao ovo, Classe 2 = ovo
classification = []
for pt in x_test.T:
    pt = pt[:, np.newaxis]
    pc1 = kernel_density_estimation(proj[:, labels == 0], pt, window)
    pc2 = kernel_density_estimation(proj[:, labels == 1], pt, window)
    if pc2 > pc1:
        classification.append(1)
    else:
        classification.append(0)

ic(time.time() - t1)
acc = np.sum(classification == y_test) / len(y_test)
ic(acc)
