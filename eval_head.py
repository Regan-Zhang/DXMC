import torch
import numpy as np
from models import *
from eval_utils import cluster_metric
from torch.utils.data import DataLoader, TensorDataset
from loss_utils import *
from data_utils import NeighborsDataset, mine_nearest_neighbors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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

    plt.savefig(f'TSNE/ImageNet-Dogs-p{p}-metric-{metric}-clusterkmeans.eps')
    # plt.show()
    # plt.close()


def infer(model, dataloader):
    model.eval()
    preds = []
    logits_image = []
    with torch.no_grad():
        for iter, (image) in enumerate(dataloader):
            image = image[0].cuda()
            _, logit_image_clu, _, _ = model(image, image)
            logit_image, _ = model.forward_embedding(image)
            pred = torch.argmax(logit_image_clu, dim=1).cpu().numpy()
            preds.append(pred)
            logits_image.append(logit_image.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    logits_image = np.concatenate(logits_image, axis=0)
    # from concat_kmeans import kmeans
    # preds = kmeans(logits_image, 15)
    return preds, logits_image


def inference(model, loader):
    model.eval()
    feature_vector = []
    for step, image in enumerate(loader):
        image = image[0].cuda()
        with torch.no_grad():
            _, z = model.forward_embedding(image)
        z = z.detach()
        feature_vector.extend(z.cpu().detach().numpy())
    return feature_vector


if __name__ == "__main__":
    dataset = "ImageNet-Dogs"  # ["CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-10", "ImageNet-Dogs", "DTD", "UCF101", "ImageNet"]
    if dataset == "UCF101" or dataset == "ImageNet":  # For large cluster number
        epochs = 100
        batch_size = 8192
        temperature = 5.0
    else:
        epochs = 20
        batch_size = 512
        temperature = 0.5
    topK = 50

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
    else:
        raise NotImplementedError

    nouns_embedding = np.load("./data/" + dataset + "_retrieved_nouns_embedding.npy")
    nouns_embedding = nouns_embedding / np.linalg.norm(
        nouns_embedding, axis=1, keepdims=True
    )
    images_embedding_train = np.load("./data/" + dataset + "_image_embedding_train.npy")
    images_embedding_train = images_embedding_train / np.linalg.norm(
        images_embedding_train, axis=1, keepdims=True
    )
    images_embedding_test = np.load("./data/" + dataset + "_image_embedding_test.npy")
    images_embedding_test = images_embedding_test / np.linalg.norm(
        images_embedding_test, axis=1, keepdims=True
    )
    labels_test = np.loadtxt("./data/" + dataset + "_labels_test.txt")

    # model = Network(in_dim=512, num_clusters=cluster_num).cuda()
    model = Network(in_dim=512, feature_dim=1024, num_clusters=cluster_num).cuda()
    model.load_state_dict(torch.load("checkpoint/ImageNet-Dogs.pth", map_location="cpu"))
    dataset_text_train = TensorDataset(torch.from_numpy(nouns_embedding).float())
    dataset_image_train = TensorDataset(
        torch.from_numpy(images_embedding_train).float()
    )
    dataset_image_test = TensorDataset(torch.from_numpy(images_embedding_test).float())

    try:
        indices_text = np.load(
            "./data/" + dataset + "_indices" + str(topK) + "_text.npy"
        )
        indices_image = np.load(
            "./data/" + dataset + "_indices" + str(topK) + "_image.npy"
        )
        print("Pre-computed indices loaded.")
    except:
        indices_text = mine_nearest_neighbors(nouns_embedding, topk=topK)
        indices_image = mine_nearest_neighbors(images_embedding_train, topk=topK)
        np.save(
            "./data/" + dataset + "_indices" + str(topK) + "_text.npy", indices_text
        )
        np.save(
            "./data/" + dataset + "_indices" + str(topK) + "_image.npy", indices_image
        )
        print("Please rerun the script.")
        exit()

    dataset = NeighborsDataset(
        dataset_text_train, dataset_image_train, indices_text, indices_image
    )
    dataloader_train = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    dataloader_test = DataLoader(
        dataset_image_test, batch_size=batch_size, shuffle=False, drop_last=False
    )

    preds, confidences_image = infer(model, dataloader_test)
    cluster_metric(labels_test, preds)

    for p in [20]:
        tsne(confidences_image, preds, p)
