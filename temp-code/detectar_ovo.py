import numpy as np
from DensityEstimation import kernel_density_estimation


def temOvo(base, x_train_c1, x_train_c2, image) -> bool:
    """
    INPUTS:

    base - base ortonormal do PCA onde os dados são projetados
    x_train_c1 - dados de treino de classe 1 (não_ovo)
    x_train_c2 - dados de treino de classe 2 (ovo)
    image - imagem que será testada

    OUTPUT:
    booleana - True se contem ovo, False caso contrário
    """

    h = 0.6
    proj = (image.ravel().reshape(1, -1) @ base).T
    assert (
        proj.shape[0] == 5
    ), f"fiz merda na hora de achatar a imagem (dims pos projecção erradas {proj.shape})"
    proj = proj.reshape(5, 1)
    pc1 = kernel_density_estimation(x_train_c1, proj, h)
    pc2 = kernel_density_estimation(x_train_c2, proj, h)
    if pc2 > pc1:
        return True
    else:
        return False
