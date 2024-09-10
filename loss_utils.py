import torch
import torch.nn.functional as F
from torch import nn


def entropy(logit):
    logit = logit.mean(dim=0)
    logit_ = torch.clamp(logit, min=1e-9)
    b = logit_ * torch.log(logit_)
    return -b.sum()


def consistency_loss(anchors, neighbors):
    b, n = anchors.size()
    similarity = torch.bmm(anchors.view(b, 1, n), neighbors.view(b, n, 1)).squeeze()
    ones = torch.ones_like(similarity)
    consistency_loss = F.binary_cross_entropy(similarity, ones)

    return consistency_loss


class DistillLoss(nn.Module):
    def __init__(self, class_num, temperature):
        super(DistillLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.mask = self.mask_correlated_clusters(class_num).cuda()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        c = F.normalize(c, dim=1)
        sim = c @ c.T / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels) / N

        return loss


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size).cuda()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class HSD(nn.Module):
    def __init__(self, tau=0.5):
        super(HSD, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.tau = tau

    def forward(self, teacher_inputs, inputs, normalized=True):
        n = inputs.size(0)

        if normalized:
            inputs = torch.nn.functional.normalize(inputs, dim=1)
            teacher_inputs = torch.nn.functional.normalize(teacher_inputs, dim=1)

        x1 = torch.pow(teacher_inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_t = x1 + x1.t()
        dist_t.addmm_(teacher_inputs, teacher_inputs.t(), beta=1, alpha=-2)
        dist_t = dist_t.clamp(min=1e-12).sqrt()  # for numerical stability

        # Compute pairwise distance
        x1 = torch.pow(teacher_inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        x2 = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = x1 + x2.t()
        dist.addmm_(teacher_inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        negative_index = (dist_t > torch.diag(dist).expand(n, n).t()).float()
        negative = dist * negative_index
        negative[negative_index == 0] = 1e5
        positive_index = 1 - negative_index
        positive = dist * positive_index

        dist_an = torch.min(negative, dim=1)
        dist_ap = torch.max(positive, dim=1)

        an_t = torch.gather(dist_t, 1, dist_an.indices.unsqueeze(1)).squeeze()
        ap_t = torch.gather(dist_t, 1, dist_ap.indices.unsqueeze(1)).squeeze()

        weight_an = torch.clamp_min(an_t.detach() - dist_an.values.detach(), min=0.)
        weight_ap = torch.clamp_min(dist_ap.values.detach() - ap_t.detach(), min=0.)

        weight_dist_an = weight_an * dist_an.values / self.tau
        weight_dist_ap = weight_ap * dist_ap.values / self.tau

        logits = torch.cat([weight_dist_an.unsqueeze(-1), weight_dist_ap.unsqueeze(-1)], dim=1)
        labels = torch.zeros(weight_dist_an.shape[0], dtype=torch.long).cuda()

        return self.loss(logits, labels)
