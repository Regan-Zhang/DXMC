import clip
import torch
from torch import nn
from timm.models.layers import trunc_normal_
from torch.nn.functional import normalize


class CLIPModel(nn.Module):
    def __init__(self, model_name="ViT-B/32"):
        super().__init__()
        self.clip, self.preprocess = clip.load(model_name, device="cuda")

    @property
    def dtype(self):
        return self.clip.visual.conv1.weight.dtype

    def encode_image(self, image):
        image_features = self.clip.visual(image.type(self.dtype))
        return image_features

    def encode_text(self, text):
        x = self.clip.token_embedding(text).type(self.dtype)

        x = x + self.clip.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip.text_projection

        return x


class ClusterHead(nn.Module):
    def __init__(self, in_dim=512, num_clusters=10):
        super().__init__()
        self.num_clusters = num_clusters
        self.cluster_head_text = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_clusters),
            nn.Softmax(dim=1),
        )
        self.cluster_head_image = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_clusters),
            nn.Softmax(dim=1),
        )
        trunc_normal_(self.cluster_head_text[0].weight, std=0.02)
        trunc_normal_(self.cluster_head_text[3].weight, std=0.02)
        trunc_normal_(self.cluster_head_image[0].weight, std=0.02)
        trunc_normal_(self.cluster_head_image[3].weight, std=0.02)

    def forward(self, text, image):
        logit_text = self.cluster_head_text(text)
        logit_image = self.cluster_head_image(image)
        return logit_text, logit_image

    def forward_embedding(self, image):
        embedding = self.cluster_head_image[0](image)
        embedding = self.cluster_head_image[1](embedding)
        embedding = self.cluster_head_image[2](embedding)
        embedding = self.cluster_head_image[3](embedding)
        return embedding


class Network(nn.Module):
    def __init__(self, in_dim=512, feature_dim=1024, num_clusters=10):
        super().__init__()
        self.num_clusters = num_clusters
        self.cluster_head_text = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_clusters),
            nn.Softmax(dim=1),
        )
        self.cluster_head_image = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_clusters),
            nn.Softmax(dim=1),
        )
        trunc_normal_(self.cluster_head_text[0].weight, std=0.02)
        trunc_normal_(self.cluster_head_text[3].weight, std=0.02)
        trunc_normal_(self.cluster_head_image[0].weight, std=0.02)
        trunc_normal_(self.cluster_head_image[3].weight, std=0.02)

        self.instance_head_text = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            # nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, feature_dim),
        )
        self.instance_head_image = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            # nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, feature_dim),
        )

    def forward(self, text, image):
        logit_text = self.cluster_head_text(text)
        logit_image = self.cluster_head_image(image)
        # logit_text_ins = normalize(self.instance_head_text(text),dim=1)
        # logit_image_ins = normalize(self.instance_head_image(image),dim=1)
        logit_text_ins = self.instance_head_text(text)
        logit_image_ins = self.instance_head_image(image)
        return logit_text, logit_image, logit_text_ins, logit_image_ins

    def forward_embedding(self, image):
        clu_embedding = self.cluster_head_image[0](image)
        clu_embedding = self.cluster_head_image[1](clu_embedding)
        clu_embedding = self.cluster_head_image[2](clu_embedding)
        clu_embedding = self.cluster_head_image[3](clu_embedding)

        ins_embedding = self.instance_head_image(image)
        return clu_embedding, ins_embedding
