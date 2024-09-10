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
import time

def tsne(x, y, epoch, p):
    metric = 'euclidean'  # cosine  euclidean   correlation
    x = TSNE(n_components=2, perplexity=p, n_iter=3000, metric=metric).fit_transform(x)
    plt.figure(figsize=(8,8))
    # ImageNet-10
    # colours = ListedColormap(
    #     ['#ECAAD7', '#FF4B4B', '#966E5A', '#969696', '#C3C33C', '#FFB050', '#64C8DC', '#3E8CBE', '#41AA41',
    #      '#A078C8'])
    # ImageNet-dogs
    colours = ListedColormap(
        ['#ECAAD7', '#FF4B4B', '#966E5A', '#969696', '#C3C33C', '#FFB050', '#64C8DC', '#3E8CBE', '#41AA41',
         '#A078C8', '#F8F819', '#636304', '#66FF66', '#333333', '#FF9999'])

    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=colours, s=1, marker='.')

    plt.savefig(f'TSNE/ImageNet-Dogs-epoch{epoch}-p{p}-metric-{metric}.eps')
    # plt.show()
    # plt.close()


def infer(model, dataloader):
    model.eval()
    preds = []
    logits_image = []
    with torch.no_grad():
        for iter, (image) in enumerate(dataloader):
            image = image[0].cuda()
            _, logit_image = model(image, image)
            pred = torch.argmax(logit_image, dim=1).cpu().numpy()
            preds.append(pred)
            logits_image.append(logit_image.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    logits_image = np.concatenate(logits_image, axis=0)
    return preds, logits_image


def inference(model, loader, class_num):
    model.eval()
    feature_vector = []
    for step, image in enumerate(loader):
        image = image[0].cuda()
        with torch.no_grad():
            _, z = model.forward_embedding(image)
        z = z.detach()
        feature_vector.extend(z.cpu().detach().numpy())
    from sklearn.cluster import KMeans
    labels_vector = KMeans(n_clusters=class_num).fit_predict(np.array(feature_vector))
    return labels_vector


if __name__ == "__main__":
    dataset = "AID"  # ["CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-10", "ImageNet-Dogs", "DTD", "UCF101", "ImageNet"]
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
    elif dataset == "Tiny-ImageNet":
        cluster_num = 200
    elif dataset == "CUB-200":
        cluster_num = 200
    elif dataset == "AID":
        cluster_num = 30
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

    model = ClusterHead(in_dim=512, num_clusters=cluster_num).cuda()
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
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99))
    distill_loss = DistillLoss(class_num=cluster_num, temperature=temperature)

    print("Start training...")
    for epoch in range(epochs):
        start = time.time()
        model.train()
        loss_distill_epoch = loss_consist_epoch = loss_entropy_epoch = 0
        for iter, (text, image, neigh_text, neigh_image) in enumerate(dataloader_train):
            text = text[0].cuda()
            image = image[0].cuda()
            neigh_text = neigh_text[0].cuda()
            neigh_image = neigh_image[0].cuda()

            logit_text, logit_image = model(text, image)
            neigh_logit_text, neigh_logit_image = model(neigh_text,neigh_image)

            loss_distill = distill_loss(logit_image, neigh_logit_text) + distill_loss(
                logit_text, neigh_logit_image
            )

            loss_consist = consistency_loss(logit_text, logit_image)
            loss_entropy = entropy(logit_text) + entropy(logit_image)

            loss = loss_distill + loss_consist - 5 * loss_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_distill_epoch += loss_distill.item()
            loss_consist_epoch += loss_consist.item()
            loss_entropy_epoch += loss_entropy.item()

            if (iter + 1) % 50 == 0 or iter + 1 == len(dataloader_train):
                print(
                    "[Epoch {}/{}] [Iter {}/{}] Loss Distill: {:.4f} Loss Consist: {:.4f} Loss Entropy: {:.4f}".format(
                        epoch + 1,
                        epochs,
                        iter + 1,
                        len(dataloader_train),
                        loss_distill.item(),
                        loss_consist.item(),
                        loss_entropy.item(),
                    )
                )
        print(
            "[Epoch: {}] Loss Cluster: {:.4f} Loss Consist: {:.4f} Loss Entropy: {:.4f}".format(
                epoch + 1,
                loss_distill_epoch / (iter + 1),
                loss_consist_epoch / (iter + 1),
                loss_entropy_epoch / (iter + 1),
            )
        )
        # print("Cluster Head Result:")
        # torch.save(model.state_dict(),f'checkpoint/ImageNet-Dogs-{epoch}.pth')
        preds, confidences_image = infer(model, dataloader_test)
        cluster_metric(labels_test, preds)

        print(f"Epoch[{epoch+1}] training takes time {time.time() - start} s")

