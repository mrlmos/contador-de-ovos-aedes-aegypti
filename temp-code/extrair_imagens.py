import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os


def isRightDim(img, dims: tuple):
    img = np.array(img)
    if img.shape == dims:
        return True
    else:
        return False


def getImage(file: str) -> np.ndarray:
    img = cv.imread(file, cv.IMREAD_GRAYSCALE)


def getImagesFromDir(path: str) -> (np.ndarray, np.ndarray):
    images = []
    labels = []
    for file in os.listdir(path):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = cv.imread(os.path.join(path, file), cv.IMREAD_GRAYSCALE)
            if isRightDim(img, (41, 41)):
                images.append(img)
                if file.startswith("nao"):
                    labels.append(0.0)
                else:
                    labels.append(1.0)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels


if __name__ == "__main__":

    # WARN: Mudar o diretório (diferente para cada um)
    path = (
        "/home/murilo/Documents/Ufs/2024.1/recpad/U1-projeto/dados/ovos-mosquito/geral"
    )

    images, labels = getImagesFromDir(path)
    timg = images[labels == 1][2]
    gx, gy = np.gradient(timg)
    timg2 = np.abs(gx) + np.abs(gy)
    # plt.imshow(timg, cmap="gray")
    plt.imshow(timg2, cmap="gray")
    plt.show()
