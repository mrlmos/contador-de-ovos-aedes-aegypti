import numpy as np
from icecream import ic
from extrair_imagens import getImagesFromDir
import matplotlib.pyplot as plt
from parzen import parzen, rect_window


def decomp(x, n):
    U, S, Vh = np.linalg.svd(x)
    S[-n:] = np.zeros(n)
    return U @ np.diag(S) @ Vh

    # min = np.min(data)
    # max = np.max(data)
    # return (data - min) / (max - min)


def normalizeImages(images):
    images = images.astype(np.float64)
    for k in range(images.shape[0]):
        min = np.min(images[k, :])
        max = np.max(images[k, :])
        images[k, :] = (images[k, :] - min) / (max - min)
    return images


def fisherLD(imagens, labels, regularizacao=False):
    c1 = imagens[labels == 1]
    c2 = imagens[labels == 0]

    mean_c1 = np.mean(c1, axis=0)
    mean_c2 = np.mean(c2, axis=0)
    cov_c1 = np.dot((c1 - mean_c1).T, (c1 - mean_c1)) / c1.shape[0]
    cov_c2 = np.dot((c2 - mean_c2).T, (c2 - mean_c2)) / c2.shape[0]
    Sw = cov_c1 + cov_c2

    if regularizacao:
        Sw = decomp(Sw, 1556)
        reg = 0.4
        Sw = reg * np.eye(Sw.shape[0]) + Sw

    w = np.linalg.inv(Sw) @ (mean_c2 - mean_c1)
    w = w / np.linalg.norm(w)
    return w, Sw


def plot_proj(w, img, labels, title: str):
    proj = np.dot(w.reshape(1, -1), img.T)
    plt.hist(proj[0, labels == 1], density=True, alpha=0.5, bins=30)
    plt.hist(proj[0, labels == 0], density=True, alpha=0.5, bins=30)
    plt.title(title)


def parzen_test(dir, w):
    imgs, labels = getImagesFromDir(dir)
    imgs = np.reshape(imgs, (199, -1))
    proj = np.dot(w.reshape(1, -1), imgs.T).reshape(-1, 1)
    return proj, labels


def main():
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    # gerando Sw pela base de jhonny
    path = "/home/murilo/Documents/Ufs/2024.1/recpad/U1-projeto/dados/base-jhonny/"
    imagens, labels = getImagesFromDir(path)
    imagens = imagens.reshape(imagens.shape[0], -1)

    w, Sw = fisherLD(imagens, labels)
    w.tofile("pesos.csv", format="csv")

    print("Sw é inversível: " + str(np.linalg.det(Sw) != 0))
    plot_proj(w, imagens, labels, "projeção na base de treino (jhonny)")
    lda = np.dot(w[np.newaxis, :], imagens.T)

    x = np.arange(-2, 2, 0.01)
    pc1 = parzen(x, lda[0, labels == 1], 0.15)
    pc2 = parzen(x, lda[0, labels == 0], 0.15)

    plt.plot(x, pc1, color="blue", linestyle="--", label=r"$p(x|c = C1)$")
    plt.plot(x, pc2, c="red", linestyle="--", label=r"$p(x|c = C2)$")

    # carregando a base pessoal e testando usando o vetor w gerado
    path = (
        "/home/murilo/Documents/Ufs/2024.1/recpad/U1-projeto/dados/ovos-mosquito/geral"
    )
    tproj, test_labels = parzen_test(path, w)
    plt.hist(tproj[test_labels == 1, 0], density=True, alpha=1.0, bins=30)
    plt.hist(tproj[test_labels == 0, 0], density=True, alpha=1.0, bins=30)
    plt.legend()
    plt.show()

    acc = []
    h_range = np.arange(0.01, 1, 0.01)
    for h in h_range:
        tlbl = []
        for point in tproj:
            if parzen(point, lda[0, labels == 1], h) > parzen(
                point, lda[0, labels == 0], h
            ):
                tlbl.append(1)
            else:
                tlbl.append(0)

        tlbl = np.array(tlbl)
        acc.append(np.sum(tlbl == test_labels) / len(test_labels))

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    ax.plot(h_range, acc)
    ax.set_title(f"melhor janela: {h_range[np.argmax(acc)]}")
    ax.set_xlabel("tamanho da janela h")
    ax.set_ylabel("acc")
    plt.show()


if __name__ == "__main__":
    main()
