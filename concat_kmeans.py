import torch
import faiss
import warnings
import numpy as np
import torch.nn.functional as F
from eval_utils import cluster_metric
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

warnings.simplefilter("ignore")


def tsne(x, y, p):
    metric = 'euclidean'  # cosine  euclidean   correlation
    # x = TSNE(n_components=2, perplexity=p, n_iter=3000, metric=metric).fit_transform(x)
    x = TSNE(n_components=2, perplexity=p, n_iter=3000, metric=metric).fit_transform(x)
    plt.figure(figsize=(8, 8))
    # ImageNet-10
    # colours = ListedColormap(
    #     ['#ECAAD7', '#FF4B4B', '#966E5A', '#969696', '#C3C33C', '#FFB050', '#64C8DC', '#3E8CBE', '#41AA41',
    #      '#A078C8'])
    # ImageNet-dogs
    colours = ListedColormap(
        ['#ECAAD7', '#FF4B4B', '#966E5A', '#969696', '#C3C33C', '#FFB050', '#64C8DC', '#3E8CBE', '#41AA41',
         '#A078C8', '#F8F819', '#636304', '#66FF66', '#333333', '#FF9999'])

    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=colours, s=1, marker='.')

    plt.savefig(f'TSNE/CLIP-ImageNet-Dogs-p{p}-metric-{metric}.eps')
    # plt.show()
    # plt.close()


def kmeans(X, cluster_num):
    print("Perform K-means clustering...")
    d = X.shape[1]
    kmeans = faiss.Kmeans(d, cluster_num, gpu=True, spherical=True, niter=300, nredo=20)
    X = X.astype(np.float32)
    kmeans.train(X)
    D, I = kmeans.index.search(X, 1)
    I = I.reshape(-1)
    print("K-means clustering done.")
    return I


if __name__ == "__main__":
    dataset = "AID"  # ["CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-10", "ImageNet-Dogs", "DTD", "UCF101", "ImageNet"]
    tau = 0.005
    if dataset == "CIFAR-10" or dataset == "STL-10" or dataset == "ImageNet-10":
        cluster_num = 10
    elif dataset == "CIFAR-20":
        cluster_num = 20
    elif dataset == "ImageNet-Dogs":
        cluster_num = 15
    elif dataset == "DTD":
        cluster_num = 47
    elif dataset == "UCF101":
        cluster_num = 101
    elif dataset == "ImageNet":
        cluster_num = 1000
    elif dataset == "CUB-200":
        cluster_num = 200
    elif dataset == "AID":
        cluster_num = 30
    else:
        raise NotImplementedError

    images_embedding = np.load("./data/" + dataset + "_image_embedding_test.npy")
    images_embedding = images_embedding / np.linalg.norm(
        images_embedding, axis=1, keepdims=True
    )
    labels = np.loadtxt("./data/" + dataset + "_labels_test.txt")

    nouns_embedding = np.load("./data/" + dataset + "_filtered_nouns_embedding.npy")
    nouns_embedding = nouns_embedding / np.linalg.norm(
        nouns_embedding, axis=1, keepdims=True
    )

    nouns_embedding = torch.from_numpy(nouns_embedding).cuda().half()
    nouns_num = nouns_embedding.shape[0]
    images_embedding = torch.from_numpy(images_embedding).cuda().half()
    image_num = images_embedding.shape[0]

    retrieval_embeddings = []
    batch_size = 8192
    for i in range(image_num // batch_size + 1):
        start = i * batch_size
        end = start + batch_size
        if end > image_num:
            end = image_num
            images_batch = images_embedding[start:end]
        similarity = torch.matmul(images_embedding[start:end], nouns_embedding.T)
        similarity = torch.softmax(similarity / tau, dim=1)
        retrieval_embedding = (similarity @ nouns_embedding).cpu()
        retrieval_embeddings.append(retrieval_embedding)
        if i % 50 == 0:
            print(f"[Completed {i * batch_size}/{image_num}]")
    retrieval_embedding = torch.cat(retrieval_embeddings, dim=0).cuda().half()
    retrieval_embedding = F.normalize(retrieval_embedding, dim=1).cpu().numpy()
    images_embedding = images_embedding.cpu().numpy()
    # print(images_embedding.shape)
    concat_embedding = np.concatenate([images_embedding, retrieval_embedding], axis=1)
    # print(concat_embedding.shape)
    preds = kmeans(concat_embedding, cluster_num)
    cluster_metric(labels, preds)
    # for p in [20]:
    #     tsne(images_embedding, preds, p)
