import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from icecream import ic


def gaussian(x: np.ndarray) -> float:
    """
    Kernel Gaussiano de covariância unitária
    """
    dim = x.shape[0]
    return (1 / ((2 * np.pi) ** (dim / 2))) * np.exp(-0.5 * np.dot(x.T, x))


def kernel_density_estimation(data: np.ndarray, points, h: float):
    """
    Estima o valor da densidade de probabilidade condicional num valor "points" dado um conjunto de treino

    INPUTS:
    data - dados de treinamento usados para estimar a densidade de probabilidade
    points - valor(es) onde a densidade de probabilidade encontrada será avaliada
    h - tamanho do janelamento do kernel

    OUTPUT:
    den - valor(es) de densidade avaliada(s) no(s) ponto(s)
    """
    dim, n_samples = data.shape
    assert (
        dim == points.shape[0]
    ), f"train data and test must have equal dimensions ( {dim} != {points.shape[0]} )"

    eval = []
    for idc, point in enumerate(points.T):
        x = point.reshape(-1, 1) - data
        eval.append(np.sum(np.apply_along_axis(gaussian, 0, x / h)))
    eval = np.array(eval)
    den = (1 / (n_samples * h)) * eval
    return den


def main():
    data = np.random.multivariate_normal(mean=[0, 0, 0], cov=np.eye(3), size=30).T
    data2 = np.random.multivariate_normal(mean=[3, 3, 3], cov=np.eye(3), size=10).T
    data = np.hstack((data, data2))
    ic(data.shape)

    r = np.linspace(-8, 8, 20)
    xgrid, ygrid, zgrid = np.meshgrid(r, r, r)
    vals = np.vstack((xgrid.ravel(), ygrid.ravel(), zgrid.ravel()))
    ic(vals.shape)

    g = kernel_density_estimation(data, vals, 0.7)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot_surface(xgrid, ygrid, g, cmap="viridis")
    # plt.show()


if __name__ == "__main__":
    main()
