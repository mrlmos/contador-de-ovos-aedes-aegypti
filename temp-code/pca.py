import numpy as np
from icecream import ic
from extrair_imagens import getImagesFromDir
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def PCA(imagens, ndims: int, save_base: bool = False) -> (np.ndarray, np.ndarray):
    """
    Projeta as imagens na direção de suas primeiras componentes principais

    ARGUMENTOS:
    imagens -- Matriz com N imagens achatadas
    ndims -- número de dimensões das quais os dados serão projetados

    RETORNA:
    proj -- Matriz com N imagens projetadas em ndims componentes principais
    """
    mean_image = np.mean(imagens, axis=1).reshape(-1, 1)
    X = imagens - mean_image
    Cov = (1 / X.shape[0]) * (np.dot(X.T, X))

    Evals, Evecs = np.linalg.eig(Cov)
    idc = np.argsort(Evals.real)[::-1]
    base = Evecs.real[:, idc[:ndims]]

    if save_base:
        np.save("./dados/base_pca.npy", base)
    proj = X @ base
    return proj, Evals[::-1].real


if __name__ == "__main__":
    path = "/home/murilo/Documents/Ufs/2024.1/recpad/U1-projeto/dados/base-jhonny/"
    imagens, labels = getImagesFromDir(path)
    imagens = imagens.reshape(imagens.shape[0], -1)
    proj, evals = PCA(imagens, ndims=5, save_base=True)

    c1 = proj[labels == 0]
    c2 = proj[labels == 1]

    evals = np.cumsum(evals) / np.sum(evals)
    ic(evals[-5])

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(c1[:, 0], c1[:, 1], c1[:, 2])
    # ax.scatter(c2[:, 0], c2[:, 1], c2[:, 2])
    # plt.show()
    #
    # plt.plot(evals[::-1])
    # plt.show()
