import torch
import torch.utils.data

import torch.nn as nn
import torch.nn.functional as F

from model.template import get_template


# Identity Layer
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_activation(argument):
    getter = {
            'relu': F.relu,
            'sigmoid': F.sigmoid,
            'softplus': F.softplus,
            'logsigmoid': F.logsigmoid,
            'softsign': F.softsign,
            'tanh': torch.tanh,
        }

    return getter.get(argument, 'Invalid activation')


# MLP Cluster, one per prototype
class Proto2Dto3DCluster(nn.Module):
    def __init__(self, opt, proto_feature, ttl_point, num_primitives):
        super(Proto2Dto3DCluster, self).__init__()
        self.opt = opt
        self.proto_feat = nn.Parameter(torch.from_numpy(proto_feature).float().unsqueeze(0), requires_grad=False) # Dummy var for initialization

        self.num_slaves = num_primitives
        self.num_per_slave = ttl_point // num_primitives
        self.template = [get_template(opt.template_type, device=opt.device) for _ in range(num_primitives)]
        print(f'New 2D -> 3D Cluster with prototype initialized, this cluster contains {self.num_slaves} slave(s)')

        self.slave_pool = nn.ModuleList([Proto2Dto3DSlave(opt) for _ in range(num_primitives)])

    def forward(self, x, td_feat=None):
        # Generate input random grid for each slave-mlp
        input_points = [self.template[idx].get_random_points(torch.Size((1, self.template[idx].dim, self.num_per_slave)))
                        for idx in range(self.num_slaves)]
        
        # Expand proto_feat and concatenate with input x (img_feat) : Batch * concat_feat
        batch_proto = self.proto_feat.repeat(x.shape[0], 1) if td_feat is None else td_feat

        x = torch.cat([x, batch_proto], dim=1)

        # Passing the actual input to its slaves
        if self.num_slaves == 1:
            output_points = torch.cat([self.slave_pool[idx](input_points[idx], x.unsqueeze(2)).unsqueeze(1)
                                       for idx in range(self.num_slaves)], dim=1).squeeze(1)
        else:
            output_points = torch.cat([self.slave_pool[idx](input_points[idx], x.unsqueeze(2)).unsqueeze(1)
                                       for idx in range(self.num_slaves)], dim=3).squeeze(1)

        return output_points

    def activate_prototype_finetune(self):
        self.proto_feat.requires_grad_(True)

    def update_prototype(self, new_proto):
        self.proto_feat.data = torch.cuda.FloatTensor(new_proto).cuda()

    def extract_prototype(self):
        return self.proto_feat.data.numpy()


# MLP Slaves, P slaves per Cluster
class Proto2Dto3DSlave(nn.Module):
    # Core component of TDPNet: deform a 2D grid (2 + encoder(I) + 3D_Proto) to a 3D point cloud patch
    def __init__(self, opt):
        super(Proto2Dto3DSlave, self).__init__()
        self.opt = opt
        self.bottleneck_size = opt.bottleneck_size
        self.input_size = opt.dim_template
        self.dim_output = 3
        self.hidden_neurons = opt.hidden_neurons
        self.num_layers = opt.num_layers

        print(f'New 2D -> 3D Slave initialized, hidden size {self.hidden_neurons}, num_layers {self.num_layers}, activation {opt.activation}')

        self.conv1 = nn.Conv1d(self.input_size, self.bottleneck_size, 1)
        self.conv2 = nn.Conv1d(self.bottleneck_size, self.hidden_neurons, 1)
        self.conv_list = nn.ModuleList([nn.Conv1d(self.hidden_neurons, self.hidden_neurons, 1) for _ in range(self.num_layers)])

        self.last_conv = nn.Conv1d(self.hidden_neurons, self.dim_output, 1)

        self.bn1 = nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_neurons)

        self.bn_list = nn.ModuleList([nn.BatchNorm1d(self.hidden_neurons) for _ in range(self.num_layers)])
        self.activation = get_activation(opt.activation)

    def forward(self, x, latent):
        x = self.conv1(x) + latent
        x = self.activation(self.bn1(x))
        x = self.activation(self.bn2(self.conv2(x)))
        for i in range(self.num_layers):
            x = self.activation(self.bn_list[i](self.conv_list[i](x)))
        return F.tanh(self.last_conv(x))
