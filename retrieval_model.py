import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tfm

DINOV2_ARCHS = {
    "s": 384,
    "b": 768,
    "l": 1024,
    "g": 1536,
}


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)


class DinoWrapper(nn.Module):
    """Same as the original DINO model, but with a linear layer on top and a resize to multiple of 14 in the forward pass."""

    def __init__(self, dino_size, feat_dim):
        super().__init__()
        assert dino_size in "sblg"
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", f"dinov2_vit{dino_size}14")
        if feat_dim is not None:
            self.fc = nn.Linear(DINOV2_ARCHS[dino_size], feat_dim)
            self.feat_dim = feat_dim
        else:
            self.fc = nn.Identity()
            self.feat_dim = DINOV2_ARCHS[dino_size]

    def resize_multiple_14(self, images):
        b, c, h, w = images.shape
        # DINO needs height and width as multiple of 14, therefore resize them to the nearest multiple of 14
        h = round(h / 14) * 14
        w = round(w / 14) * 14
        images = tfm.functional.resize(images, [h, w], antialias=True)
        return images

    def forward(self, images):
        images = self.resize_multiple_14(images)
        B, C, H, W = images.shape
        features = self.dinov2(images)
        features = self.fc(features)
        features = L2Norm()(features)
        return features
