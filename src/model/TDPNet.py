import torch
import torch.nn as nn
import torch.utils.data
import numpy as np

from torchvision.models import vgg16_bn
from model.net_components import Proto2Dto3DCluster

# Pending imports


class TDPNet(nn.Module):
    # Core TDPNet module
    # configuration -- The setting of network
    # prototypes -- The initialized 3D prototypes
    def __init__(self, configuration, prototypes, num_pts=2048):
        super(TDPNet, self).__init__()
        self.opt = configuration
        self.device = self.opt.device
        self.num_prototypes = prototypes.shape[0]
        self.num_slaves = self.opt.num_slaves

        self.num_pts_per_proto = num_pts // self.num_prototypes

        # Image Encoder Part
        self.img_feature_extractor = vgg16_bn(pretrained=True).features
        self._set_finetune()

        self.img_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Image feat -> 3D Decoder part. Using 1 MLP as placeholder
        self.decoder = nn.ModuleList([Proto2Dto3DCluster(self.opt, prototypes[idx], self.num_pts_per_proto, self.num_slaves)
                                      for idx in range(self.num_prototypes)])

    def forward(self, x, img_flag=True):
        if img_flag:
            # Get image feature vector -- Batch * 512
            latent_vector = self.img_pool(self.img_feature_extractor(x)).squeeze(-1).squeeze(-1)

            # Deform each Patch
            output_points = torch.cat([self.decoder[idx](latent_vector) for idx in range(self.num_prototypes)], dim=2)
        else:
            # Dummy image feature
            latent_vector = torch.cuda.FloatTensor(np.ones((x.shape[0], 512)))

            # Deform each Patch using real pc features
            output_points = torch.cat([self.decoder[idx](latent_vector, x) for idx in range(self.num_prototypes)], dim=2)

        return output_points.transpose(1, 2).contiguous()

    def _set_finetune(self):
        active_layer = 3
        for idx in range(len(self.img_feature_extractor)-1, -1, -1):
            if isinstance(self.img_feature_extractor[idx], nn.Conv2d):
                if active_layer > 0:
                    self.img_feature_extractor[idx].requires_grad_(True)
                    active_layer -= 1
                else:
                    self.img_feature_extractor[idx].requires_grad_(False)
        return None

    def activate_prototype_finetune(self):
        for idx in range(self.num_prototypes):
            self.decoder[idx].activate_prototype_finetune()
        return None

    def update_prototypes(self, prototypes):
        for idx in range(self.num_prototypes):
            self.decoder[idx].update_prototype(np.expand_dims(prototypes[idx], axis=0))

    def extract_prototypes(self):
        return np.concatenate([self.decoder[idx].extract_prototype() for idx in range(self.num_prototypes)])






